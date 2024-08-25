""" Dataset helper functions """
import random
import numpy as np
from omegaconf.listconfig import ListConfig
import nrrd
import torch
import torch.nn.functional as F
from utils.image import crop_and_pad, normalise_intensity
from utils.image_io import load_nifti
from scipy.io import loadmat,savemat
import hdf5storage


def _to_tensor(data_dict):
    # cast to Pytorch Tensor
    for name, data in data_dict.items():
        data_dict[name] = torch.from_numpy(data)
    return data_dict


def _to_ndarry(data_dict):
    # cast to Numpy array
    for name, data in data_dict.items():
        data_dict[name] = data.numpy()
    return data_dict


def _crop_and_pad(data_dict, crop_size):
    # cropping and padding
    for name, data in data_dict.items():
        data_dict[name] = crop_and_pad(data, new_size=crop_size)
    return data_dict


def _normalise_intensity(data_dict, keys, vmin=0., vmax=1.):
    """ Normalise intensity of data in `data_dict` with `keys` """

    # images in one pairing should be normalised using the same min-max
    # vmin_in = np.amin(np.array([data_dict[k] for k in keys]))
    # vmax_in = np.amax(np.array([data_dict[k] for k in keys]))

    for k, x in data_dict.items():
        if k in keys:
            dim = x.ndim - 1
            image_axes = tuple(range(1, 1 + dim))
            
            vmin_in = np.min(x, axis=image_axes, keepdims=True) #N*1*1
            vmax_in = np.max(x, axis = image_axes, keepdims=True) #N*1*1
            
            x = vmin + (vmax - vmin) * (x - vmin_in) / (vmax_in - vmin_in) 
            # data_dict[k] = normalise_intensity(x,
            #                                    min_in=vmin_in, max_in=vmax_in,
            #                                    min_out=vmin, max_out=vmax,
            #                                    mode="minmax", clip=True)
            data_dict[k] = x
    return data_dict


def _resample(data_dict, size=None, scale_factor=None):
    for k, x in data_dict.items():
        if k == 'tar_seg' or k == 'src_seg':
            align_corners = None
            mode = 'nearest'
            x = x.float()
        else:
            align_corners = True
            mode = ['bilinear', 'trilinear'][x.ndim-3]
        data_dict[k] = F.interpolate(x.unsqueeze(0),
                                     size=size, scale_factor=scale_factor,
                                     recompute_scale_factor=True if scale_factor else False,
                                     mode=mode, align_corners=align_corners)[0]
    return data_dict


def _shape_checker(data_dict):
    """Check if all data points have the same shape
    if so return the common shape, if not print data type"""
    data_shapes_dict = {n: x.shape for n, x in data_dict.items()}
    shapes = [x for _, x in data_shapes_dict.items()]
    if all([s == shapes[0] for s in shapes]):
        common_shape = shapes[0]
        return common_shape
    else:
        raise AssertionError(f'Not all data points have the same shape, {data_shapes_dict}')


def _magic_slicer(data_dict, slicing=None):
    """
    Select all slices, one random slice, or some slices within `slice_range`, according to `slicing`
    Works with ndarray
    """
    # slice selection
    #num_slices = _shape_checker(data_dict)[0] #original code is [0]
    num_slices = data_dict['tar'].shape[3]
    slice_range = (0, num_slices)
    num_slices_patch = 64

    # select slice(s)
    if slicing == 'None':
        # all slices within slice_range
        slicer = slice(slice_range[0], slice_range[1])
        # z = slice_range[0]+1
        # slicer = slice(z, z + 1)

    elif slicing == 'random':
        # randomly choose one slice within range
        z = random.randint(slice_range[0], slice_range[1]-1-num_slices_patch+1)
        slicer = slice(z, z + num_slices_patch)  # use slicer to keep dim

    elif isinstance(slicing, (list, tuple, ListConfig)):
        # slice several slices specified by slicing
        assert all(0 <= i <= 1 for i in slicing), f'Relative slice positions {slicing} need to be within [0, 1]'
        slicer = tuple(int(i * (slice_range[1] - slice_range[0])) + slice_range[0] for i in slicing)

    else:
        raise ValueError(f'Slicing mode {slicing} not recognised.')

    # slicing
    for name, data in data_dict.items():
        data_dict[name] = data[:, :, :, slicer]  # (N, H, W)

    return data_dict


def _clean_seg(data_dict, classes=(0, 1, 2, 3)):
    """ Remove (zero-fill) slices where either ED or ES frame mask is empty """
    # ndarray each of shape
    tar_seg = data_dict['tar_seg']
    src_seg = data_dict['src_seg']
    num_slices = tar_seg.shape[0]
    assert tuple(np.unique(tar_seg)) == tuple(np.unique(src_seg)) == classes
    non_empty_slices = [np.prod([np.sum((tar_seg[i] == cls) * (src_seg[i] == cls)) > 0
                                 for cls in classes]) > 0 for i in range(num_slices)]
    non_empty_slices = np.nonzero(non_empty_slices)[0]
    # slices_to_take = non_empty_slices[[len(non_empty_slices) // 2 + i for i in range(-1, 2, 1)]]
    slices_to_take = non_empty_slices
    tar_seg_out = np.zeros_like(tar_seg)
    src_seg_out = np.zeros_like(src_seg)
    tar_seg_out[slices_to_take] = tar_seg[slices_to_take]
    src_seg_out[slices_to_take] = src_seg[slices_to_take]
    data_dict['tar_seg'] = tar_seg_out
    data_dict['src_seg'] = src_seg_out
    return data_dict


def _sosimg(img, coil):
    imgsos = (coil.conj()*img).sum(axis=3)
    coilsos = (coil.conj()*coil).sum(axis=3)
    imgsc0 = imgsos/(coilsos + 1e-12)
    imgsc = np.abs(imgsc0)
    #savemat('/home/ubuntu/Share/FanYang/CoronaryImaging/test/datatest.mat', {'datasos': imgsc})
    return imgsc


def _load3d_mrca(data_path_dict, coilmap=None):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # shape (H, W, N) ->  (N, H, W)
        if 'nrrd' in data_path:
            x, _ = nrrd.read(data_path)
            data_dict[name] = x.transpose(2, 0, 1)
        elif 'mat' in data_path:
            if 'motionv4' in data_path:
                x = hdf5storage.loadmat(data_path)['dispX']  # ny*nx*nz
                x = x[:, :, 0:88]
                # x = x[:, :, 12:]
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
                x_pad = np.pad(x, ((0, 0), (1, 1), (0, 0), (0, 0)), 'edge')
                x_pad = x_pad.transpose(0, 3, 1, 2)

                y = hdf5storage.loadmat(data_path)['dispY']
                y = y[:, :, 0:88]
                # y = y[:, :, 12:]
                y = y.reshape((1, y.shape[0], y.shape[1], y.shape[2]))
                y_pad = np.pad(y, ((0, 0), (1, 1), (0, 0), (0, 0)), 'edge')
                y_pad = y_pad.transpose(0, 3, 1, 2)

                z = hdf5storage.loadmat(data_path)['dispZ']
                z = z[:, :, 0:88]
                # z = z[:, :, 12:]
                z = z.reshape((1, z.shape[0], z.shape[1], z.shape[2]))
                z_pad = np.pad(z, ((0, 0), (1, 1), (0, 0), (0, 0)), 'edge')
                z_pad = z_pad.transpose(0, 3, 1, 2)

                # xtmp = np.concatenate((x, y, z), axis=0)
                xtmp = np.concatenate((z_pad, y_pad, x_pad), axis=0)
                data_dict[name] = xtmp

            elif 'motion' in data_path:
                # x = loadmat(data_path)['disp'].transpose(3,2,0,1)
                x = hdf5storage.loadmat(data_path)['disp'].transpose(3,2,0,1) #nz*ndirection*ny*nx
                xtmp = x*1.0
                xtmp[:,0,:,:] = x[:,1,:,:]
                xtmp[:,1,:,:] = x[:,0,:,:]
                data_dict[name] = xtmp
                # savemat('/home/ubuntu/Share/FanYang/CoronaryImaging/test/motion.mat', {'motion': data_dict[name] })
            else:
                x = hdf5storage.loadmat(data_path)['imgrecon'].transpose(2,0,1)  #nz*ny*nx   #5min
                x = np.abs(x)
                x = x[0:88, :, :]
                # x = x[12:, :, :]
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
                data_dict[name] = x
                # savemat('/home/ubuntu/Share/FanYang/CoronaryImaging/test/sosdata.mat', {'imgsc': data_dict[name] })
        else:
            data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
    return data_dict

def _load3d_mrca_p2(data_path_dict, data_dict, coilmap=None):
    # data_dict = dict()
    for name, data_path in data_path_dict.items():
        # shape (H, W, N) ->  (N, H, W)
        if 'nrrd' in data_path:
            x, _ = nrrd.read(data_path)
            data_dict[name] = x.transpose(2, 0, 1)
        elif 'mat' in data_path:
            if 'motionv4' in data_path:
                x = hdf5storage.loadmat(data_path)['dispX']  # ny*nx*nz
                x = x[:, :, 0:88]
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
                x_pad = np.pad(x, ((0, 0), (1, 1), (0, 0), (0, 0)), 'edge')
                x_pad = x_pad.transpose(0, 3, 1, 2)

                y = hdf5storage.loadmat(data_path)['dispY']
                y = y[:, :, 0:88]
                y = y.reshape((1, y.shape[0], y.shape[1], y.shape[2]))
                y_pad = np.pad(y, ((0, 0), (1, 1), (0, 0), (0, 0)), 'edge')
                y_pad = y_pad.transpose(0, 3, 1, 2)

                z = hdf5storage.loadmat(data_path)['dispZ']
                z = z[:, :, 0:88]
                z = z.reshape((1, z.shape[0], z.shape[1], z.shape[2]))
                z_pad = np.pad(z, ((0, 0), (1, 1), (0, 0), (0, 0)), 'edge')
                z_pad = z_pad.transpose(0, 3, 1, 2)

                # xtmp = np.concatenate((x, y, z), axis=0)
                xtmp = np.concatenate((z_pad, y_pad, x_pad), axis=0)
                data_dict[name] = xtmp

            elif 'motion' in data_path:
                # x = loadmat(data_path)['disp'].transpose(3,2,0,1)
                x = hdf5storage.loadmat(data_path)['disp'].transpose(3,2,0,1) #nz*ndirection*ny*nx
                xtmp = x*1.0
                xtmp[:,0,:,:] = x[:,1,:,:]
                xtmp[:,1,:,:] = x[:,0,:,:]
                data_dict[name] = xtmp
                # savemat('/home/ubuntu/Share/FanYang/CoronaryImaging/test/motion.mat', {'motion': data_dict[name] })
            else:
                x = hdf5storage.loadmat(data_path)['imgrecon'].transpose(2,0,1)  #nz*ny*nx   #5min
                x = np.abs(x)
                x = x[0:88, :, :]
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
                data_dict[name] = x
                # savemat('/home/ubuntu/Share/FanYang/CoronaryImaging/test/sosdata.mat', {'imgsc': data_dict[name] })
        else:
            data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
    return data_dict


def _load2d(data_path_dict, coilmap=None):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # shape (H, W, N) ->  (N, H, W)
        if 'nrrd' in data_path:
            x, _ = nrrd.read(data_path)
            data_dict[name] = x.transpose(2, 0, 1)
        elif 'mat' in data_path:
            if 'motionv2' in data_path:
                x = hdf5storage.loadmat(data_path)['disp'].transpose(3,2,0,1) #nz*ndirection*ny*nx
                xtmp = x*1.0
                xtmp[:,0,:,:] = x[:,1,:,:]
                xtmp[:,1,:,:] = x[:,0,:,:]
                data_dict[name] = xtmp
            elif 'motion' in data_path:
                # x = loadmat(data_path)['disp'].transpose(3,2,0,1)
                x = hdf5storage.loadmat(data_path)['disp'].transpose(3,2,0,1) #nz*ndirection*ny*nx
                xtmp = x*1.0
                xtmp[:,0,:,:] = x[:,1,:,:]
                xtmp[:,1,:,:] = x[:,0,:,:]
                data_dict[name] = xtmp
                # savemat('/home/ubuntu/Share/FanYang/CoronaryImaging/test/motion.mat', {'motion': data_dict[name] })
            else:
                # # x = loadmat(data_path)['img'].transpose(2,0,1,3,4)
                # x = loadmat(data_path)['img'].transpose(2,1,0,3,4) #nz*nx*ny*nc*necho
                # xoutofphase = x[:,:,:,:,1] 
                # data_dict[name] = _sosimg(xoutofphase, coilmap)
                # x = loadmat(data_path)['imgrecon'].transpose(2,0,1) #nz*ny*nx   #fulldata
                x = hdf5storage.loadmat(data_path)['imgrecon'].transpose(2,0,1)  #nz*ny*nx   #5min
                x =  np.abs(x)
                data_dict[name] = x
                # savemat('/home/ubuntu/Share/FanYang/CoronaryImaging/test/sosdata.mat', {'imgsc': data_dict[name] })
        else:
            data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
    return data_dict


def _load2dxz(data_path_dict, coilmap=None):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # shape (H, W, N) ->  (N, H, W)
        if 'nrrd' in data_path:
            x, _ = nrrd.read(data_path)
            data_dict[name] = x.transpose(2, 0, 1)
        elif 'mat' in data_path:
            if 'motionv2' in data_path:
                x = hdf5storage.loadmat(data_path)['disp'].transpose(3,2,0,1) #ny*ndirection*nx*nz
                xtmp = x*1.0
                xtmp[:,0,:,:] = x[:,1,:,:]
                xtmp[:,1,:,:] = x[:,0,:,:]
                data_dict[name] = xtmp
            elif 'motion' in data_path:
                # x = loadmat(data_path)['disp'].transpose(3,2,0,1)
                x = hdf5storage.loadmat(data_path)['disp'].transpose(3,2,0,1) #ny*ndirection*nx*nz
                xtmp = x*1.0
                xtmp[:,0,:,:] = x[:,1,:,:]
                xtmp[:,1,:,:] = x[:,0,:,:]
                data_dict[name] = xtmp
                # savemat('/home/ubuntu/Share/FanYang/CoronaryImaging/test/motion.mat', {'motion': data_dict[name] })
            else:
                # # x = loadmat(data_path)['img'].transpose(2,0,1,3,4)
                # x = loadmat(data_path)['img'].transpose(2,1,0,3,4) #nz*nx*ny*nc*necho
                # xoutofphase = x[:,:,:,:,1] 
                # data_dict[name] = _sosimg(xoutofphase, coilmap)
                # x = loadmat(data_path)['imgrecon'].transpose(2,0,1) #nz*ny*nx   #fulldata
                # x = hdf5storage.loadmat(data_path)['imgrecon'].transpose(2,0,1)  #nz*ny*nx   #5min
                x = hdf5storage.loadmat(data_path)['imgrecon']  #ny*nx*nz   #5min
                x =  np.abs(x)
                data_dict[name] = x
                # savemat('/home/ubuntu/Share/FanYang/CoronaryImaging/test/sosdata.mat', {'imgsc': data_dict[name] })
        else:
            data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
    return data_dict


def _load3d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
        data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
    return data_dict
