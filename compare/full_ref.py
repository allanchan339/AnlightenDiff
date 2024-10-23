# NOTE: skimage.__version__ == '0.17.1'
# Example run: python ref.py --test_dir_pred /root/autodl-tmp/Result/RetinexNet/low --test_dir_gt /root/autodl-tmp/Dataset/Clean_Images/low
"""
all right reserved by https://github.com/ShenZheng2000/LLIE_Survey/blob/main/Metric/ref.py
"""
import os
import numpy as np
from glob import glob
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
import lpips
import argparse
from natsort import natsorted
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1 #normalize to [-1,1]


def _psnr(tf_img1, tf_img2):
    return compare_psnr(tf_img1, tf_img2, data_range=255)

def _ssim_gray(tf_img1, tf_img2):
    return compare_ssim(cv2.cvtColor(tf_img1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(tf_img2, cv2.COLOR_RGB2GRAY), multichannel=True, data_range=255)

def _ssim(tf_img1, tf_img2):
    # NOTE: see multichannel=True for RGB images
    return compare_ssim(tf_img1, tf_img2, multichannel=True, data_range=255)

@torch.no_grad()
def _lpips(tf_img1, tf_img2, loss_fn_alex):
    scores = loss_fn_alex(tf_img1, tf_img2)
    if scores.numel() == 1:
        return scores.item()
    else:
        return scores

def center_crop(img_real, img_fake):
    # Get the height and width of the smaller image
    h, w = img_fake.shape[:2]  
    
    # Find the center of the larger image
    center = img_real.shape[:2]  
    
    # Calculate the x and y coordinates to crop the larger image 
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    
    # Crop the larger image centered at (x, y) to size (w, h)
    img_real = img_real[int(y):int(y+h), int(x):int(x+w)]
    return img_real

def main(args):

    # NOTE: add sorted
    path_real = natsorted(glob(os.path.join(args.test_dir_gt, '*')))
    path_fake = natsorted(glob(os.path.join(args.test_dir_pred, '*')))
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    list_psnr = []
    list_ssim = []
    list_lpips = []

    for i in range(len(path_real)):
        print(f"Comparing {path_real[i]} with {path_fake[i]}")

        # read images
        # print("==========================>")
        # print("path_real[i]", path_real[i])
        # print("path_fake[i]", path_fake[i])
        img_real = cv2.imread(path_real[i])
        img_fake = cv2.imread(path_fake[i])

        if img_real.shape != img_fake.shape:
            img_real = center_crop(img_real, img_fake)

        # convert to torch tensor for lpips calculation
        tes_real = transform(img_real).to(device)
        tes_fake = transform(img_fake).to(device)

        # calculate scores
        psnr_num = _psnr(img_real, img_fake)
        ssim_num = _ssim(img_real, img_fake)
        lpips_num = _lpips(tes_real, tes_fake, loss_fn_alex)

        # append to list
        list_psnr.append(psnr_num)
        list_ssim.append(ssim_num)
        list_lpips.append(lpips_num)

        print(f"CPSNR: {psnr_num:.4f}; CSSIM: {ssim_num:.4f}; CLPIPS: {lpips_num:.4f}")

    # Average score for the dataset
    print("======={}=======>".format(args.test_dir_gt))
    print("======={}=======>".format(args.test_dir_pred))

    print("Average PSNR:", "%.3f" % (np.mean(list_psnr)))
    print("Average SSIM:", "%.3f" % (np.mean(list_ssim)))
    print("Average LPIPS:", "%.3f" % (np.mean(list_lpips)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_dir_gt', type=str,
                        default='',
                        help='directory for clean images',
                        required=True)
    parser.add_argument('--test_dir_pred', type=str,
                        default='',
                        help='directory for enhanced or restored images',
                        required=True)
    args = parser.parse_args()
    main(args)
