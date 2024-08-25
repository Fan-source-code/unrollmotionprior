import torch
from torch import nn as nn
from core.transformations import warp_fn_3D
import numpy as np
import nibabel as nib


def save_nifti(x, path, nim=None, verbose=False):
    """
    Save a numpy array to a nifti file

    Args:
        x: (numpy.ndarray) data
        path: destination path
        nim: Nibabel nim object, to provide the nifti header
        verbose: (boolean)

    Returns:
        N/A
    """
    if nim is not None:
        nim_save = nib.Nifti1Image(x, nim.affine, nim.header)
    else:
        nim_save = nib.Nifti1Image(x, np.eye(4))
    nib.save(nim_save, path)

    if verbose:
        print("Nifti saved to: {}".format(path))


class DClayer(torch.nn.Module):
    """Base regulariser class"""
    def __init__(self):
        super(DClayer, self).__init__()


    def grad(self, tar, src, disp, grid, netoutput, reguweight_):
        """ Gradient step of the SSD similairty loss"""
        disp.requires_grad_()  # this is a function, adjuct disp.requires_grad to True
        with torch.set_grad_enabled(True):
            warped_src = warp_fn_3D(src, disp[:, :, :, 1:-1, :], grid, backwards_grad=True)

            dc = (((tar - warped_src).abs()) ** 2).sum()

            regularization = (((disp - netoutput).abs()) ** 2).sum()
            cost = dc + reguweight_ * regularization

        grad = torch.autograd.grad(cost, disp, create_graph=self.training)[0]

        regularization = regularization.detach()
        dc = dc.detach()
        warped_src = warped_src.detach()
        # disp.requires_grad_(False)
        disp = disp.detach()
        grad = grad.detach()
        cost = cost.detach()
        return grad, cost

    def forward(self, dispini, tar, src, netoutput, grid, reguweight_):


        disp = dispini * 1.0

        g, cost = self.grad(tar, src, disp, grid, netoutput, reguweight_) #sample-wise gradient
        
        d = -1.0*g
        # start GD iteration: sample-wise GD
        stepsize = 0.1
        CG_steps = 10
        for ite in range(CG_steps):
            dispnew = disp + stepsize * d

            gnew, costnew = self.grad(tar, src, dispnew, grid, netoutput, reguweight_)

            if costnew > cost:
                ite = ite - 1
                if stepsize > 1e-4:
                    stepsize = stepsize/2
                    continue
                else:
                    break

            cost = costnew*1.0
            d = -1.0*gnew
            disp = dispnew*1.0

        return disp
