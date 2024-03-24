import torch
import numpy as np

def index(feat, uv):
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

def index_3d(feat, XYZ):
    """
    input
        feat: (C,B,D,H,W) 3d deepVoxels' features in unit cube [0,1]^3
        XYZ : (B,3,N), 3d coords. for tri-linear sampling
    return
        samples: (B,C,N) sampled features from deepVoxels
    """

    # init.
    depthVoxels  = feat.shape[2] # 32
    heightVoxels = feat.shape[3] # 48
    widthVoxels  = feat.shape[4] # 32

    
    
    # normalize into torch.float32 of W~(0,32), H~(0,48), D~(0,32)
    X = XYZ[:, 0, :] # (B, N)
    Y = XYZ[:, 1, :] # (B, N)
    Z = XYZ[:, 2, :] # (B, N)
    X = X * (widthVoxels-1.)
    Y = Y * (heightVoxels-1.)
    Z = Z * (depthVoxels-1.)

    # get min, max of torch.int64
    x0 = X.floor().long() # (B, N)
    x1 = (x0 + 1).long()  # (B, N)
    y0 = Y.floor().long() # (B, N)
    y1 = (y0 + 1).long()  # (B, N)
    z0 = Z.floor().long() # (B, N)
    z1 = (z0 + 1).long()  # (B, N)

    # clip into valid ranges, torch.int64
    x0 = torch.clamp(x0, min=0, max=widthVoxels  - 1) # (B, N)
    x1 = torch.clamp(x1, min=0, max=widthVoxels  - 1)
    y0 = torch.clamp(y0, min=0, max=heightVoxels - 1)
    y1 = torch.clamp(y1, min=0, max=heightVoxels - 1)
    z0 = torch.clamp(z0, min=0, max=depthVoxels  - 1)
    z1 = torch.clamp(z1, min=0, max=depthVoxels  - 1)

    # compute offset, torch.float32
    x_ = X - x0.float() # (B, N)
    y_ = Y - y0.float()
    z_ = Z - z0.float()

    # get idx for batch
    ix = torch.zeros_like(x0) # (B,N)
    for j in range(ix.shape[0]): ix[j] += j

    # tri-linear interpolation, (C,B,N)
    out = (feat[:, ix, z0, y0, x0]*(1-x_)*(1-y_)*(1-z_) +
           feat[:, ix, z1, y0, x0]*x_*(1-y_)*(1-z_) +
           feat[:, ix, z0, y1, x0]*(1-x_)*y_*(1-z_) +
           feat[:, ix, z0, y0, x1]*(1-x_)*(1-y_)*z_ +
           feat[:, ix, z1, y0, x1]*x_*(1-y_)*z_ +
           feat[:, ix, z0, y1, x1]*(1-x_)*y_*z_ +
           feat[:, ix, z1, y1, x0]*x_*y_*(1-z_) +
           feat[:, ix, z1, y1, x1]*x_*y_*z_)

    # (B,C,N)
    return out.transpose(0,1).contiguous()

def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz
