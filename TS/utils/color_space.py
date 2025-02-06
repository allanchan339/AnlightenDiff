from utils.module_util import normalize_to_neg_one_to_one,  unnormalize_to_zero_to_one
import torch
from torch.nn.functional import l1_loss

def img_normalize(img_hr, img_lr, rescale_ratio=1.0, clip_input=False, autonormalize=True):
    x = img_hr 

    if autonormalize:
        x = normalize_to_neg_one_to_one(x)
    
    if clip_input:
        x = x.clamp(-1,1)
    return x


def img2res(img_hr, img_lr, rescale_ratio=1.0, clip_input=False, autonormalize=True):
        # both img_hr and img_lr in [0,1]
        x = (img_hr - img_lr) * rescale_ratio
        
        if autonormalize:
            x = normalize_to_neg_one_to_one(x)
        if clip_input:
            x = x.clamp(-1,1)
        return x
    

def img_unnormalize(img_pred, img_lr, rescale_ratio=1.0, clip_input=True,  autonormalize=True):
    img_pred = img_pred
    
    if clip_input:
        img_pred = img_pred.clamp(-1,1)

    if autonormalize:
        img_pred = unnormalize_to_zero_to_one(img_pred)
    
    return img_pred

def res2img(res, img_lr, rescale_ratio=1.0, clip_input=True,  autonormalize=True):
        # img_lr in [0,1] ; res in [-1,1]
        res = res / rescale_ratio  # unscale to decrease sensitivity
        if clip_input:
            res = res.clamp(-1,1) # force to -1,1
        
        if autonormalize:
            res = unnormalize_to_zero_to_one(res) # return to 0,1
        
        img = img_lr + res # add image

        if clip_input:
            img = img.clamp(0,1) # to solve overflow
        return img

def rgb2hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    https://github.com/limacv/RGB_HSV_HSL
    """
    assert rgb.min() >= 0 and rgb.max() <= 1
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.
    hsl_h /= 6.

    hsl_l = (cmax + cmin) / 2.
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
    return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)

def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

def hslimg2res(img_hr, img_lr, clip_input=False, rescale_ratio = 2.0, autonormalize=True):
    """
    img_hr and img_lr in RGB and in [0,1] 
    convert to hsl and take res, then normalize
    """
    img_hr = rgb2hsl_torch(img_hr)
    img_lr = rgb2hsl_torch(img_lr)

    x = (img_hr - img_lr)

    if autonormalize:
        x = normalize_to_neg_one_to_one(x)
    
    x *= rescale_ratio

    if clip_input:
        x = x.clamp(-1,1)

    return x

def hslres2img(res, img_lr, clip_input= True, rescale_ratio = 2.0, autonormalize=True):
    res = res / rescale_ratio  # unscale to decrease sensitivity
    
    if clip_input:
        res = res.clamp(-1,1) # force to -1,1
    
    if autonormalize:
        res = unnormalize_to_zero_to_one(res)  # return to 0,1

    img_lr = rgb2hsl_torch(img_lr)

    img = img_lr + res # in hsl space
    
    if clip_input:
        img = img.clamp(0,1) # to solve overflow

    img = hsl2rgb_torch(img)

    if clip_input:
        img = img.clamp(0,1) # to solve overflow
    
    return img



if __name__ == '__main__':
    input = torch.rand((1,3,256,256))
    print('rgb:', input.min(), input.max())

    hsl = rgb2hsl_torch(input)
    print('h:',hsl[0,0].min(), hsl[0,0].max())
    print('s:',hsl[0,1].min(), hsl[0,1].max())
    print('l:',hsl[0,2].min(), hsl[0,2].max())

    rgb = hsl2rgb_torch(hsl)

    print('loss', l1_loss(rgb, input))
