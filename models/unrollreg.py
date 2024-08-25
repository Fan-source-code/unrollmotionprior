import torch

# import matplotlib.pyplot as plt
# from torch.nn import functional as F
from core.transformations import warp, warp_fn
# from core.utils import create_img_pyramid
# from core.utils import interpolate_nd
from models.base import LitDLReg
# from utils.visualise import visualise_seq_results

def finite_difference(x):
    x = x.squeeze()  # 3*nz*ny*nx
    fd_1 = x[:, 1:, :, :] - x[:, :-1, :, :]
    fd_2 = x[:, :, 1:, :] - x[:, :, :-1, :]
    fd_3 = x[:, :, :, 1:] - x[:, :, :, :-1]

    reg_fd = ((fd_1.abs())**2).sum() + ((fd_2.abs())**2).sum() + ((fd_3.abs())**2).sum()  # l2norm
    # reg_fd = (fd_1.abs()).sum() + (fd_2.abs()).sum() + (fd_3.abs()).sum()  # l1norm
    # reg_fd = (fd_1*(fd_1.conj())).sum() + (fd_2*(fd_2.conj())).sum() + (fd_3*(fd_3.conj())).sum()
    return reg_fd

class LitUnrollReg(LitDLReg):
    def __init__(self, *args, **kwargs):
        super(LitUnrollReg, self).__init__(*args, **kwargs)

    def inference(self, batch):

        tars = [batch['tar']]
        srcs = [batch['src']]

        out, zs, respme_out = self.forward(tars, srcs)
        return {'disp': out, 'tars': tars, 'srcs': srcs, 'z': zs, 'respme_out': respme_out}

    @staticmethod
    def get_norm_grid(size):
        grid = torch.meshgrid([torch.linspace(-1, 1, s) for s in size])
        grid = torch.stack(grid, 0).requires_grad_(False)  # (ndims, *size)
        return grid

    def loss_fn(self, outputs, batch):
        respme_out = outputs['respme_out']
        tar = outputs['tars'][-1]
        src = outputs['srcs'][-1]
        disp = outputs['disp'][-1]
        disp_label = batch['disp_gt'].squeeze().view(disp.shape)
        z = outputs['z'][-1]

        size = tar.shape
        grid = self.get_norm_grid(size[2:])
        grid = grid.to(device=tar.device)

        warped_src = warp_fn(src, disp[:, :, :, 1:-1, :], grid)
        losses = {}
        reguweight_ = self.network.regu_weight

        costdc = self.train_loss_fn(tar, warped_src)
        costreg = self.train_loss_fn(disp, z)
        losses['cost'] = costdc+reguweight_*costreg
        losses['cost_dc'] = costdc
        losses['cost_reg'] = costreg

        warped_src_respme = warp_fn(src, respme_out[:, :, :, 1:-1, :], grid)
        dc_respme = self.train_loss_fn(tar, warped_src_respme)
        reg_respme = finite_difference(respme_out[:, :, :, 1:-1, :])/(size[2]*size[3]*size[4])
        loss_mse = self.train_loss_fn(disp, disp_label)
        losses['dc_initial'] = dc_respme
        losses['reg_initial'] = reg_respme
        losses['label_mse'] = loss_mse
        loss = loss_mse + dc_respme + 0.001*reg_respme
        losses['train_loss'] = loss

        return loss, losses
