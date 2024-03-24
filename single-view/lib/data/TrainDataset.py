from torch.utils.data import Dataset
import numpy as np
import os
import random
import math
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import time
#from .lib.smpl_util import SMPLX, load_fit_body
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign

log = logging.getLogger('trimesh')
log.setLevel(40)
def build_triangles(vertices, faces):

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device=vertices.device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]
    
def make_rotate(rx, ry, rz):
    '''
    Used for rotating the 3dmm mesh to certain angle,
    as we just keep the frontal view mesh in the preprocessed dataset
    for better alignment and saving disk storage
    '''
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def load_obj_mesh(mesh_file, with_normal=False, with_color=False):
    
    tdmmpt_vertex = []
    tdmmpt_norm = []
    vertex_color_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            # coordinates only: v x y z
            v = list(map(float, values[1:4]))
            tdmmpt_vertex.append(v)
            # coordinates & color: v x y z r g b
            # if len(values) == 7 :
            vc = list(map(float, values[4:7]))
                # Append 1 for Alpha in RGBA
                # vc = list(map(float, values[4:7]+[1]))
            vertex_color_data.append(vc)
            
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            tdmmpt_norm.append(vn)

    vertices = np.array(tdmmpt_vertex)
    vertex_color = np.array(vertex_color_data)
    
    if with_normal:
        norms = np.array(tdmmpt_norm)
        norms = normalize_v3(norms)
        return vertices, norms
    if with_color:
        return vertices, vertex_color

        
    return vertices

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

def sample_surface_with_normals(mesh, num_sample_inout) :
    """ Sample points from mesh surface along with the normals
    
    Ref: https://github.com/mikedh/trimesh/issues/1285#issuecomment-880986179
    """
    # Get mesh faces
    faces = mesh.faces
    # Get mesh triangle vertices (shape: (n,3,3))
    triangles = mesh.vertices[faces]
    
    # Sample points on mesh surface
    samples, fid = mesh.sample(num_sample_inout, return_index=True)
    
    # compute the barycentric coordinates of each sample
    bary = trimesh.triangles.points_to_barycentric(
        triangles=triangles[fid], points=samples)
    # interpolate vertex normals from barycentric coordinates
    interp = trimesh.unitize((mesh.vertex_normals[faces[fid]] *
                              trimesh.unitize(bary).reshape(
                                  (-1, 3, 1))).sum(axis=1))

    return samples, interp

import open3d as o3d
def get_signed_distance(mesh_path, query_point) :
    """ Get signed distance from mesh.
    
    Ref.: http://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html
    
    Args:
    - mesh_path: (str) path to mesh file.
    - query_point: (np.ndarray) query points, shape: (N,3)
    
    Returns:
    - signed_distance: (np.ndarray) signed distances from mesh, shape: (N)
    
    """
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    query_point = o3d.core.Tensor(query_point, dtype=o3d.core.Dtype.Float32)
    
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
    
    # Compute distance of the query point from the surface
    signed_distance = scene.compute_signed_distance(query_point) # shape: (N)
    
    return signed_distance.numpy()

def get_unsigned_distance(mesh_path, query_point) :
    """ Get unsigned distance from mesh.
    
    Ref.: http://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html
    
    Args:
    - mesh_path: (str) path to mesh file.
    - query_point: (np.ndarray) query points, shape: (N,3)
    
    Returns:
    - unsigned_distance: (np.ndarray) unsigned distances from mesh, shape: (N)
    
    """
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    query_point = o3d.core.Tensor(query_point, dtype=o3d.core.Dtype.Float32)
    
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
    
    # Compute distance of the query point from the surface
    unsigned_distance = scene.compute_distance(query_point)
    
    return unsigned_distance.numpy()


class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.root_512 = self.opt.dataroot_512
        
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.SMPLX = os.path.join(self.root, 'SMPLX')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')
        
        ## BBox for samples randomly in image space
        self.B_MIN = np.array([-0.5, -0.5, -0.5])
        self.B_MAX = np.array([0.5, 0.5, 0.5])
        
        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize
        self.num_views = self.opt.num_views
        
        # number of sample points
        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.yaw_list = list(range(0,360,1))
        self.pitch_list = [0]
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])
        # NOTE: we choose to load trimesh during the training procedure
#        self.mesh_dic = load_trimesh(self.OBJ)

    def get_subjects(self):
        all_subjects = os.listdir(self.SMPLX)
        var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_render_shape(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img_1024': [num_views, C, W, H] high-res images
            'img_512': [num_views, C, W, H] low-res images
            'calib_1024': [num_views, 4, 4] calibration matrix for high-res images
            'calib_512': [num_views, 4, 4] calibration matrix for high-res images
            'face_region': [num_views, 1, 4] region to locate face in 3D space
            'face_tdmm': [num_views, n, 3] 3dmm vertex points
            'view_id': [num_views, 1, 1] view angle for rotating the 3D points
        '''
        pitch = self.pitch_list[pid]
        
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)] for offset in range(num_views)]
        
        # The ids are an even distribution of num_views around view_id
        view_ids = [view_ids[(yid + len(view_ids) // num_views * offset) % len(view_ids)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)
        
        render_list = []
        calib_list = []
        view_id = []
        
        smpl_list = []
        smpl_face_list = []
        smpl_normal_list = []

        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center')
            # model rotation
            R = param.item().get('R')

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)
                    
                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))

                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.load_size // 2)
                trans_intrinsic[1, 3] = -dy / float(self.load_size // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)
                    render_512 = render_512.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            
            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            
            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            
            # smpl
            
            smpl_path = os.path.join(self.SMPLX, subject, 'mesh_smplx.obj')
            smpl_obj = trimesh.load(smpl_path)
            # smpl_vertex = torch.tensor(smpl_obj.vertices).unsqueeze(0).float()
            smpl_vertex = torch.tensor(smpl_obj.vertices).permute(1, 0)
            smpl_face = torch.tensor(smpl_obj.faces)
            
            angle = int(vid)
            # R_smpl =  torch.from_numpy(make_rotate(0, math.radians(int(vid)), 0)).float()
            R_smpl = torch.from_numpy(np.matmul(make_rotate(math.radians(0), 0, 0), make_rotate(0, math.radians(angle), 0)))
            smpl_vertex = torch.mm(R_smpl, smpl_vertex)
            smpl_vertex = smpl_vertex.permute(1, 0).float()
            
            # rotate_vertex = smpl_vertex.squeeze(0).float()
            # rotate_vertex = torch.mm(rotate_vertex, R_smpl.T)
            # smpl_obj = trimesh.Trimesh(vertices = rotate_vertex, faces = smpl_face)
            smpl_obj = trimesh.Trimesh(vertices = smpl_vertex, faces = smpl_face)
            smpl_vertex = torch.tensor(smpl_obj.vertices).unsqueeze(0).float()
            smpl_face = torch.tensor(smpl_obj.faces)
            smpl_normal = torch.tensor(smpl_obj.vertex_normals)
            
            smpl_face_list.append(smpl_face)
            smpl_list.append(smpl_vertex)
            smpl_normal_list.append(smpl_normal)
                    
            view_id.append(R_smpl)
        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'smpl': torch.stack(smpl_list, dim=0),
            'smpl_face': torch.stack(smpl_face_list, dim=0),
            'smpl_normal': torch.stack(smpl_normal_list, dim=0),
            'view_id': torch.stack(view_id, dim=0)
        }

    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        mesh_path = os.path.join(self.OBJ, subject, subject + '.obj')
        mesh = trimesh.load(os.path.join(self.OBJ, subject, subject + '.obj'))
        mesh_vertices = torch.from_numpy(mesh.vertices).unsqueeze(0)
        mesh_faces = torch.from_numpy(mesh.faces).unsqueeze(0)
        
        surface_points, surface_faces = trimesh.sample.sample_surface(mesh, self.num_sample_inout)
        
        surface_normals = mesh.face_normals[surface_faces]
        
        samples_pre, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
        sample_points = samples_pre + np.random.normal(scale=self.opt.sigma, size=samples_pre.shape)

        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        
        surface_samples, _ = trimesh.sample.sample_surface(mesh, self.num_sample_inout)
        
        sample_points = np.concatenate([sample_points, random_points], 0)
        
        np.random.shuffle(sample_points)
        
        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]
        

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]
        
        sample_points = np.concatenate([inside_points, outside_points], 0).T
        surface_points = surface_points.T
        surface_normals = surface_normals.T
        sample_distance = get_unsigned_distance(mesh_path, sample_points.T)
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)
        sample_sdf = sample_distance * (labels.astype(np.float32)*2 - 1)*(-1)
        
        samples_points = torch.Tensor(sample_points).float()
        surface_points = torch.Tensor(surface_points).float()
        surface_normals = torch.Tensor(surface_normals).float()
        labels = torch.Tensor(labels).float()
        sample_sdf = torch.Tensor(sample_sdf).float()
        
#        save_samples_truncted_prob('out.ply', samples.T, labels.T)
#        exit()
        del mesh

        return {
            'samples': samples_points,
            'surface': surface_points,
            'normals': surface_normals,
            'labels': labels,
            'sample_sdf': sample_sdf
        }
        
    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject+'.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }

        if self.opt.num_sample_inout:
            render_data = self.get_render_shape(subject, num_views=self.num_views, yid=yid, pid=pid, random_sample=self.opt.random_multiview)
            res.update(render_data)
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)

        return res

    def __getitem__(self, index):
        import cv2
        cv2.setNumThreads(0)
        return self.get_item(index)
