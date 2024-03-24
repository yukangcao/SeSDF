import torch
import torch.nn as nn
import numpy as np
import trimesh
import torch.nn.functional as F
from torch.autograd import grad
import math
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier, SurfaceClassifier_joint
from .DepthNormalizer import DepthNormalizer
from .MeshNormalizer import MeshNormalizer
from .HGFilters import *
from .PointNetFilters import LocalPoolPointnet
from ..net_util import init_net
from ..mesh_util import cal_sdf
from .common import normalize_3d_coordinate
from .embedder import *

def make_rotate(rx, ry, rz):
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
    
def gradient(inputs, outputs):

    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    
    print(inputs.shape, outputs.shape, d_points.shape)
    
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        allow_unused=True,
        retain_graph=True,
        only_inputs=True)
    print(points_grad)
    print(points_grad[1].shape)

#    [0][:, -3:]
    
    return points_grad
    
class HGPIFuNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'hgpifu'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)
        self.smpl_filter = LocalPoolPointnet()
        
        self.sdf_clip = 15.0 / 100.0
                                
        self.surface_classifier_occ = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim_occ,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())
            
        self.surface_classifier_sdf = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim_sdf,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())
        
        self.normalizer = DepthNormalizer(opt)
        self.MeshNormalizer = MeshNormalizer()
        
        embed_fn, input_ch = get_embedder(6)
        self.embed_fn = embed_fn
        
        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.im_smpl_feat_list = []
        self.tmpx = None
        self.normx = None
        
        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        # print(self.im_feat_list[-1].cpu().detach().numpy().shape)
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
            
    def filter_smpl(self, smpl):
        '''
        Filter the input tdmm mesh
        '''
        self.im_smpl_grid_feats = self.smpl_filter(smpl)
            

    def query(self, points, calibs, smpl_vertex, smpl_face, smpl_normal, transforms=None, py_min=None, py_max=None, view_id=None, surface=False):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :param face_region: [B, 1, 4] face region in 2d space
        :param py_min & py_max: [1] value for rescaling the sample points to [-0.5,0.5]
        :param view_id: angle for rotating sample points to index 3D features
        
        :return: [B, Res, N] predictions for each point
        '''
        
        batch_size = points.size()[0]
        
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        
        z_feat = self.normalizer(z, calibs=calibs)
        
        # list for storing predictions
        if surface:
            self.intermediate_preds_surface_list = []
            self.intermediate_preds_surface_grad_list = []
        else:
            self.intermediate_preds_list = []
            self.intermediate_preds_samples_list = []
            self.intermediate_preds_samples_grad_list = []
            
        points_xy = points[:, :2, :]
        points_z = points[:, 2:3, :]
        
        # normalize the sample points to [-0.5, 0.5]
        points_scale = points.clone()
        points_scale[:, 1, :] = points_scale[:, 1, :] - py_min
        points_scale = points_scale / (py_max - py_min)
        points_scale[:, 1, :] = points_scale[:, 1, :] - 0.5
        
        points_rotate = []
        # rotate the points to certain angle
        # we won't rotate points when testing
        if view_id != None:
            view_id = view_id.squeeze(1)
            for i, view in enumerate(view_id):
                R = view_id[i].float()
                rotate_vertex = points_scale[i, :, :].float()
                rotate_vertex = torch.mm(R, rotate_vertex)
                rotate_vertex = rotate_vertex.unsqueeze(0).float()
                points_rotate.append(rotate_vertex)
            points_scale = torch.cat(points_rotate)
                        
        points_z = points[:, 2:3, :]
        points_norm = normalize_3d_coordinate(points_scale, padding=0.1)
        points_num = points.size()[-1]
        
        sdf_points = points_norm.clone()
        sdf_points = sdf_points.permute(0, 2, 1).contiguous()
        sdf_smpl, normal_smpl = cal_sdf(smpl_vertex.float(), smpl_face.squeeze(0), smpl_normal.squeeze(0), sdf_points.detach().float())
        sdf_smpl = sdf_smpl.permute(0, 2, 1)
        normal_smpl = normal_smpl.permute(0, 2, 1).float()
                
        sdf_smpl = self.normalizer(sdf_smpl, calibs=calibs)
        sdf_smpl = self.embed_fn(sdf_smpl.permute(0,2,1))
        sdf_smpl = sdf_smpl.permute(0, 2, 1)
        
        for im_feat in self.im_feat_list:
            # original 2d features
            point_local_feat_list = [self.index(im_feat, xy), z_feat]
            point_local_feat = torch.cat(point_local_feat_list, 1)

            # add 3d feature
            smpl_feats = self.im_smpl_grid_feats

            points_norm_grid = points_norm.clone()
            points_norm_grid -= 0.5
            points_norm_grid *= 2
            
            interp_smpl_3dfeats = F.grid_sample(smpl_feats, points_norm_grid.permute(0, 2, 1).unsqueeze(2).unsqueeze(2), mode='bilinear')
            point_feat_3d = interp_smpl_3dfeats.view([batch_size, -1, points_num])
            
            point_local_feat3d_list = [point_feat_3d, sdf_smpl]
            point_local_feat3d = torch.cat(point_local_feat3d_list, 1)
            
            point_local_feat2d = point_local_feat
            
            point_local_feat_all = torch.cat([point_local_feat2d, point_feat_3d, sdf_smpl, normal_smpl], 1)
            sdf_module = self.surface_classifier_sdf(point_local_feat_all)
            
            if surface:
                self.intermediate_preds_surface_list.append(sdf_module[:, 0, :])
                self.intermediate_preds_surface_grad_list.append(sdf_module[:, 1:, :])
            else:
                sdf_field = sdf_module[:, 0, :].unsqueeze(0)
                sdf_field = self.embed_fn(sdf_field.permute(0,2,1))
                sdf_field = sdf_field.permute(0, 2, 1).repeat(self.num_views, 1, 1)
                
                normal_field = sdf_module[:, 1:, :].repeat(batch_size, 1, 1)
                
                point_local_feat_field = torch.cat([point_local_feat2d, point_feat_3d, sdf_field, normal_smpl], dim=1)
                pred_occ = self.surface_classifier_occ(point_local_feat_field)
                
                self.intermediate_preds_list.append(pred_occ)
                self.intermediate_preds_samples_list.append(sdf_module[:, 0, :])
                self.intermediate_preds_samples_grad_list.append(sdf_module[:, 1:, :])
                
                self.preds = self.intermediate_preds_list[-1]
        
    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]
        
    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error_occ = 0
        error_sdf = 0
        error_grad = 0
        error_normal = 0
        error_sdf_s = 0
        
        for preds in self.intermediate_preds_surface_list:
            error_sdf += (preds.abs()).mean()
        error_sdf /= len(self.intermediate_preds_surface_list)
                
        for preds in self.intermediate_preds_samples_list:
            error_sdf_s += (preds - self.labels_sdf).mean()
        error_sdf_s /= len(self.intermediate_preds_samples_list)
        
        for preds in self.intermediate_preds_samples_grad_list:
            error_grad += ((preds.norm(2, dim=-1)-1)**2).mean()
        error_grad /= len(self.intermediate_preds_samples_grad_list)
        
        for preds in self.intermediate_preds_surface_grad_list:
            error_normal += ((preds - self.labels_normal).abs()).norm(2, dim=1).mean()
        error_normal /= len(self.intermediate_preds_surface_grad_list)
        
        for preds in self.intermediate_preds_list:
            error_occ += self.error_term(preds, self.labels_occ)
        error_occ /= len(self.intermediate_preds_list)
        
        error = 0.15 * error_sdf + 0.15 * error_sdf_s + 0.1 * error_normal + error_occ + 0.1 * error_grad
        
        return error

    def forward(self, images, points, surface, calibs, smpl_vertex=None, smpl_face=None, smpl_normal=None, transforms=None, labels=None, normals=None, sdf=None, view_id=None):
    
        self.labels_occ = labels
        self.labels_normal = normals
        self.labels_sdf = sdf
        
        # Phase 1: get the 2D and 3D features
        # Get image feature
        self.filter(images)
        
        # Get the SMPL feature
        py_min, py_max, smpl_vertices_norm = self.MeshNormalizer(smpl_vertex)
        smpl_vertex = smpl_vertices_norm.squeeze(0).permute(1, 0)
        smpl_vertex_norm = normalize_3d_coordinate(smpl_vertex, padding=0.1)
            
        smpl_vertex_norm = smpl_vertex_norm.unsqueeze(0)
        self.filter_smpl(smpl_vertex_norm)
            
        # Phase 2: point query
        self.query(points=points.float(), calibs=calibs, smpl_vertex=smpl_vertex_norm, smpl_face=smpl_face, smpl_normal=smpl_normal, transforms=transforms, py_min=py_min, py_max=py_max, view_id=view_id, surface=False)
        
        self.query(points=surface.float(), calibs=calibs, smpl_vertex=smpl_vertex_norm, smpl_face=smpl_face, smpl_normal=smpl_normal, transforms=transforms, py_min=py_min, py_max=py_max, view_id=view_id, surface=True)
        
        # get the prediction
        res = self.get_preds()
            
        # get the error
        error = self.get_error()

        return res, error
