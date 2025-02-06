import argparse
import torch
import torch.nn.functional as F


def mergeConfig(*args):
    tmp = argparse.Namespace()
    for i in args:
        tmp = argparse.Namespace(**tmp.__dict__, **i.__dict__)
    return tmp


def average_brightness(imgs):
    return torch.mean(imgs)

def pad_to_multiple(tensor, mul=32):
    B, C, H, W = tensor.size()

    pad_H = (mul - H % mul) % mul
    pad_W = (mul - W % mul) % mul

    tensor = F.pad(tensor, (0, pad_W, 0, pad_H), mode='reflect')

    return tensor

def unpad_from_multiple(padded_tensor, original_shape):

    B, C, H, W = padded_tensor.size()
    B_orig, C_orig, H_orig, W_orig = original_shape

    if H < H_orig or W < W_orig:
        raise ValueError("Padded tensor has smaller H/W than original")

    pad_H = H - H_orig
    pad_W = W - W_orig
    
    if pad_H == 0 and pad_W == 0:
        return padded_tensor
    elif pad_H == 0:
        unpadded = padded_tensor[:, :, :, 0:-pad_W]
    elif pad_W == 0:
        unpadded = padded_tensor[:, :, 0:-pad_H, :]
    else:
        unpadded = padded_tensor[:, :, 0:-pad_H, 0:-pad_W]

    assert unpadded.shape == original_shape
    return unpadded
