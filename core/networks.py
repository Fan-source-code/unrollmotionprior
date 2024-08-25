import torch
import torch.nn as nn

from core.reg_terms import CNNRegulayer

from core.utils import interpolate_nd, convNd
from core.dc_terms import DClayer
from torch.nn import functional as F
from core.layers import VecInt

    
    
class UnrollReg(nn.Module):
    r"""
    Unrolling Based Image Registration Network

    `num_blocks` is a list of number of blocks in each resolution.
    The dc and regulariser blocks are each a nn.ModuleList of nn.ModuleList(s),
    each list for one resolution:
    `[
    [block1, block2, ..., block_<num_blocks[0]>],
    [block1, block2, ..., block_<num_blocks[1]>],
    ...]`
    """
    def __init__(self,
                 ndim,
                 size,
                 unroll_iterations,
                 regu_weight,
                 regulariser_config
                 ):
        super(UnrollReg, self).__init__()

        self.ndim = ndim

        self.num_resolutions = 1


        self.regu_weight = regu_weight

        self.unroll_iterations = unroll_iterations

        self.regulariser_config = regulariser_config


        self.dc = DClayer()
        self.REGULARISER = CNNRegulayer(self.regulariser_config)
        self.motion = UNetmotion()


    @staticmethod
    def get_norm_grid(size):
        grid = torch.meshgrid([torch.linspace(-1, 1, s) for s in size])
        grid = torch.stack(grid, 0).requires_grad_(False)  # (ndims, *size)
        return grid
    
    def get_theta(self):
        """ return all parameters of the regularization """
        return self.named_parameters()

    def forward(self, tars: list, srcs: list):
        """" Input `tars` and `srcs` are list of images with increasing resolution """
        # initialise disp
        respme_input = torch.cat([srcs[0], tars[0]], 1)
        flag = False
        if respme_input.shape[2] == 100:
            flag = True
            respme_input = F.pad(respme_input, (0, 0, 3, 3, 2, 2), "constant", 0)
        else:
            respme_input = F.pad(respme_input, (0, 0, 3, 3), "constant", 0)
        [z, invz] = self.motion(respme_input)
        if flag:
            z = z[:, :, 2:-2, 2:-2, :]
        else:
            z = z[:, :, :, 2:-2, :]
        respme_out = z*1.0
        device = tars[0].device

        disps = []
        zs = []
        size = tars[0].shape
        grid = self.get_norm_grid(size[2:])
        grid = grid.to(device=device)
        for lvl in range(self.num_resolutions):
            tar, src = tars[lvl], srcs[lvl]

            for idxiter in range(self.unroll_iterations):
                

                dispini = z*1.0
                disp = self.dc(dispini, tar, src, z, grid, self.regu_weight)  # contain 3 GD steps

                z_input = F.interpolate(disp, scale_factor=0.25, mode='trilinear', align_corners=True)

                z_out = self.REGULARISER(z_input)

                z = F.interpolate(z_out, scale_factor=4, mode='trilinear', align_corners=True)
                
                disps.append(disp)
                zs.append(z)


        return zs, disps, respme_out

class UNetmotion(nn.Module):
    r"""
    Adapted from the U-net used in VoxelMorph:
    https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """
    def __init__(self,
                 ndim=3,
                 enc_channels=(32, 64, 128),
                 dec_channels=(256, 128, 64),
                 # enc_channels=(64, 128, 256),
                 # dec_channels=(128, 64),
                 out_channels=(1, 1),
                 conv_before_out=False
                 ):
        super(UNetmotion, self).__init__()
        int_steps = 7
        # down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(int_steps)

        self.ndim = ndim

        self.input_layers = nn.ModuleList()
        self.input_layers.append(
            nn.Sequential(
                convNd(ndim, 2, enc_channels[0], stride=1, a=0.2),
                # nn.ReLU(inplace=True)
                nn.LeakyReLU(0.2)
            )
        )

        # self.input_layers = nn.Conv3d(4, enc_channels[0], 3, stride=1, padding=1)


        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            # in_ch = 4 if i == 0 else enc_channels[i - 1]
            stride = 1
            self.enc.append(
                nn.Sequential(
                    convNd(ndim, enc_channels[i], enc_channels[i], stride=stride, a=0.2),
                    # nn.ReLU(inplace=True)
                    nn.LeakyReLU(0.2)
                )
            )

        # downsampler
        # self.downsample = nn.MaxPool3d(2)
        # self.downsample = nn.AvgPool3d(4)
        self.downsample = nn.ModuleList()
        for i in range(int(len(enc_channels))):
            stride = 1
            self.downsample.append(
                nn.Sequential(
                    # nn.MaxPool3d(2, padding=0, return_indices=True),
                    nn.MaxPool3d(2),
                    convNd(ndim, enc_channels[i], enc_channels[i]*2, stride=stride, a=0.2),
                    # nn.ReLU(inplace=True)
                    nn.LeakyReLU(0.2)
                )
            )


        # decoder layers
        self.dec = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(len(dec_channels)):
            in_ch = dec_channels[i] if i == 0 else dec_channels[i - 1]
            self.dec.append(
                nn.Sequential(
                    convNd(ndim, in_ch, dec_channels[i], a=0.2),
                    # nn.ReLU(inplace=True)
                    nn.LeakyReLU(0.2)
                )
            )
            self.upsample.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='trilinear'),
                    convNd(ndim, dec_channels[i], int(dec_channels[i]/2), a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )


        # (optional) conv layers before prediction
        if conv_before_out:
            self.out_layers = nn.ModuleList()
            for i in range(len(out_channels)):
                in_ch = dec_channels[-1] if i == 0 else out_channels[i-1]
                self.out_layers.append(
                    nn.Sequential(
                        convNd(ndim, in_ch, out_channels[i], a=0.2),  # stride=1
                        nn.LeakyReLU(0.2)
                    )
                )

            # final prediction layer with additional conv layers
            self.out_layers.append(
                convNd(ndim, out_channels[-1], ndim)
            )

        else:

            # final prediction layer without additional conv layers
            self.out_layers = nn.ModuleList()
            self.out_layers.append(
                convNd(ndim, dec_channels[-1], ndim*1) #3directions*3bins
            )

    def forward(self, xinput):

        x = xinput*1.0

        # encoder
        fm_enc = [x]
        fm_enc.append(self.input_layers[0](fm_enc[-1]))
        for i, enc in enumerate(self.enc):
            fm_enc.append(enc(fm_enc[-1]))  # 6 data in total
            fm_enc.append(self.downsample[i](fm_enc[-1]))


        # decoder: conv + upsample + concatenate skip-connections (to full resolution)
        dec_out = fm_enc[-1]
        for i, dec in enumerate(self.dec):
            dec_out = dec(dec_out)
            dec_out = self.upsample[i](dec_out)
            dec_out = torch.cat([dec_out, fm_enc[-2 - i - i]], dim=1)

        # further convs and prediction
        pos_flow = dec_out
        for out_layer in self.out_layers:
            pos_flow = out_layer(pos_flow)

        neg_flow = -pos_flow
        pos_flow_bin1 = self.integrate(pos_flow[:, 0:3, :, :, :])  # dispz, dispy, dispx


        neg_flow_bin1 = self.integrate(neg_flow[:, 0:3, :, :, :])

        return [pos_flow_bin1, neg_flow_bin1]
