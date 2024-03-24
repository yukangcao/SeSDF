import torch
import numpy as np
import trimesh
from .mesh_util import *
from .sample_util import *
from .geometry import *
import cv2
from PIL import Image
from tqdm import tqdm
#from .model.MeshNormalizer import MeshNormalizer


def MeshNormalizer(tdmm_vertices):
    py_min = -1.0
    py_max = 1.0
    tdmm_vertices_norm = tdmm_vertices
    tdmm_vertices_norm = tdmm_vertices_norm.permute(0, 2, 1)

    tdmm_vertices_norm[:, 1, :] = tdmm_vertices_norm[:, 1, :] - py_min
    tdmm_vertices_norm = tdmm_vertices_norm / (py_max - py_min)
    tdmm_vertices_norm[:, 1, :] = tdmm_vertices_norm[:, 1, :] - 0.5
        
    return py_min, py_max, tdmm_vertices_norm
    
def normalize_3d_coordinate(p, padding=0.1):
    '''
    Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor
    
def reshape_multiview_tensors(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def gen_mesh(opt, netG, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
        
    smpl_tensor = data['smpl'].to(device=cuda)
    smpl_face_tensor = data['smpl_face'].to(device=cuda).unsqueeze(0)
    smpl_normal_tensor = data['smpl_normal'].to(device=cuda).unsqueeze(0)
    netG.filter(image_tensor)
        
    py_min, py_max, smpl_vertices_norm = MeshNormalizer(smpl_tensor)
    smpl_vertex_norm = normalize_3d_coordinate(smpl_vertices_norm.squeeze(0).permute(1, 0), padding=0.1)
    smpl_vertex = smpl_vertex_norm.unsqueeze(0).float()
    netG.filter_smpl(smpl_vertex)
    
    
    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)
        verts, faces, _, _ = reconstruction(
            netG, cuda, calib_tensor, smpl_vertex, smpl_face_tensor, smpl_normal_tensor, opt.resolution, b_min, b_max, py_min, py_max, use_octree=use_octree)
            
        # get PIFu's color
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        xyz_tensor = netG.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
        
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def calc_error(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            surface_tensor = data['surface'].to(device=cuda).unsqueeze(0)
            
            smpl_tensor = data['smpl'].to(device=cuda).squeeze(0)
            smpl_face_tensor = data['smpl_face'].to(device=cuda).unsqueeze(0)
            smpl_normal_tensor = data['smpl_normal'].to(device=cuda).unsqueeze(0)
            view_id_tensor = data['view_id'].to(device=cuda).unsqueeze(0)
            
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
                surface_tensor = reshape_sample_tensor(surface_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)
            normal_tensor = data['normals'].to(device=cuda).unsqueeze(0)
            sdf_tensor = data['sample_sdf'].to(device=cuda).unsqueeze(0)
            
            res, error = net.forward(image_tensor, sample_tensor, surface_tensor, calib_tensor, smpl_vertex=smpl_tensor, smpl_face=smpl_face_tensor, smpl_normal=smpl_normal_tensor, labels=label_tensor, normals=normal_tensor, sdf=sdf_tensor, view_id=view_id_tensor)

            IOU, prec, recall = compute_acc(res, label_tensor)

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)
    
    return np.average(error_color_arr)

