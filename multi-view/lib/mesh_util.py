from skimage import measure
import numpy as np
import torch
import trimesh
from .sdf import create_grid, eval_grid_octree, eval_grid
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from pytorch3d.structures import Meshes

def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights
    
def build_triangles(vertices, faces):

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device=vertices.device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]

def cal_sdf(verts, faces, normals, points):
    # functions modified from ICON
    
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    
    Bsize = points.shape[0]
    
    triangles = build_triangles(verts, faces)
    normals_tri = build_triangles(normals, faces)
    
    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    
    closest_normals = torch.gather(normals_tri, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    closest_triangles = torch.gather(
        triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    bary_weights = barycentric_coordinates_of_projection(
        points.view(-1, 3), closest_triangles)
    
    pts_normals = (closest_normals*bary_weights[:, None]).sum(1).unsqueeze(0)
    
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1, 1), pts_normals.view(Bsize, -1, 3)
    
def feat_select(feat, select):

    # feat [B, featx2, N]
    # select [B, 1, N]
    # return [B, feat, N]

    dim = feat.shape[1]
    num_views= feat.shape[0]
        
    feat *= select
    
    feat_select = torch.divide(torch.sum(feat, 0, keepdim=True), torch.sum(select, 0, keepdim=True))

    return feat_select
    
def reconstruction(net, cuda, calib_tensor, smpl_tensor, smpl_vertex_norm, smpl_face_tensor, smpl_normal,
                   resolution, b_min, b_max, py_min=None, py_max=None,
                   use_octree=False, num_samples=10000, transform=None): # return verts, faces, normals, values
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.  
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param face_region_tensor: face region bbox
    :param tdmm_tensor: providing tdmm_vertex for sdf calculation
    :param tdmm_face_tensor: providing tdmm_faces for sdf calculation
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max,
                              transform=transform)
    smpl_tensor = smpl_tensor.squeeze(0).float()
                
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    # Then we define the lambda function for cell evaluation
    def eval_func(points, py_min, py_max):
        points = np.expand_dims(points, axis=0) # 在axis=0的位置加一个维度
        points = np.repeat(points, net.num_views, axis=0)  # 延y轴复制num_view次数
        samples = torch.from_numpy(points).to(device=cuda).float() # 转化至tensor，flaot数
        
        depth = []
        for i in range(3):
            samples_i = samples[i, :, :].permute(1,0).cpu()
            smpl_vi = smpl_tensor[i, :, :].cpu()
            smpl_depth = smpl_vi[:, 2:3]
            smpl_fi = smpl_face_tensor.squeeze(0)[i, :, :].cpu()
            smpl_obj = trimesh.Trimesh(vertices=smpl_vi, faces=smpl_fi)
            rmi = RayMeshIntersector(smpl_obj, scale_to_box=False)
            
            ray_dirs = np.zeros_like(samples_i)
            ray_dirs[:, 2] = -1.0
                        
            hit_face_samples = rmi.intersects_first(ray_origins=samples_i, ray_directions=ray_dirs)
            hit_vertices_samples = smpl_fi[hit_face_samples, :]
            hit_depth_samples = smpl_depth[hit_vertices_samples[:, 0], 0].unsqueeze(0).unsqueeze(0)
            depth.append(hit_depth_samples)
        depth = torch.cat(depth, dim=1).to(device=cuda).float()
        
        net.query(samples, calib_tensor, smpl_vertex_norm, smpl_face_tensor, smpl_normal, depth, transforms=transform, py_min=py_min, py_max=py_max, surface=False)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy() # detach阻断反向传播，返回tensor；cup（）把变量放置cpu上，返回tensor；numpy（）转换为numpy

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, py_min, py_max, num_samples=num_samples) # 运用octree时的eval_func的运用
    else:
        sdf = eval_grid(coords, eval_func, py_min, py_max, num_samples=num_samples) # 不运用octree时的eval_func的运用
    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5) # verts: spatial coordinates for V unique mesh vertices. faces: triangular faces (each face has exactly three indices). normals: normal direction at each vertec. values: maximum value of the data in the local region near each vertex.
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w') # 打开mesh路径

    for v in verts: # 记录verts
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces: # 记录faces
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close() #关闭


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w') #打开mesh路径

    for idx, v in enumerate(verts):
        c = colors[idx] # 第idx个点的color值
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces: # f_plus
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w') # 打开mesh路径

    for idx, v in enumerate(verts):
        vt = uvs[idx] # 第idx个点的uv值
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1 # 0,0, 2,2, 1,1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
