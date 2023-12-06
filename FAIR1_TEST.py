from model.mymodel import BlindSR
import torch
import numpy as np
import imageio
import argparse
import os
import utility
import cv2
import math
import skimage.io as io
import PIL.Image as pil_image
import matplotlib.pyplot as plt
from util import calculate_psnr,calculate_ssim
from scipy import ndimage as ndi
from skimage import feature
import lpips
loss_fn_alex = lpips.LPIPS(net='alex')
import skimage.measure
# skimage.measure.compare_psnr(sr_block, hr_block, data_range=255)
# skimage.measure.compare_ssim(sr_block, hr_block, data_range=255 ,multichannel=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str,
                        default='/home/damon/Downloads/DASR-main/remote_sensing/Data/WHU-RS19/HR',
                        help='image directory')
    parser.add_argument('--scale', type=str, default='2',
                        help='super resolution scale')
    parser.add_argument('--resume', type=int, default=694,
                        help='resume from specific checkpoint')
    parser.add_argument('--blur_type', type=str, default='iso_gaussian',
                        help='blur types (iso_gaussian | aniso_gaussian)')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.blur_type == 'iso_gaussian':
        dir = '/home/damon/Downloads/DASR-main/experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_iso'
    elif args.blur_type == 'aniso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_aniso'

    # path to save sr images
    # save_dir = dir + '/results/WHU-RS19/0-SR'
    save_dir = '/media/damon/0B79054F0B79054F/DSR_2'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    DASR = BlindSR(args).cuda()
    DASR.load_state_dict(torch.load(dir + '/model/model_AID_MSR_E3_x2' + str(args.resume) + '.pt'), strict=False)
    DASR.eval()
    eval_psnr = 0
    eval_ssim = 0
    eval_lpips = 0

    LR_dir = '/media/damon/0B79054F0B79054F/images (1)'
    string = args.img_dir + '/*.jpg'
    coll = io.ImageCollection(string)
    LR_coll = io.ImageCollection(LR_dir + '/*.tif')

    for i in range(len(LR_coll)):

        lr = pil_image.open(LR_coll.files[i]).convert('RGB')
        # if lr.width * lr.height > pow(3000, 2):
        #
        # else:
        lr = np.array(lr).astype(np.uint8)
        lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
        lr = torch.from_numpy(lr).float().cuda().unsqueeze(0).unsqueeze(0)

        # inference
        sr = DASR(lr[:, 0, ...])
        sr = utility.quantize(sr, 255.0)

        # save sr results
        img_name = LR_coll.files[i].replace(LR_dir + '/', '')
        # img_name = img_name.replace('low', 'answer')
        # img_name = img_name.replace('jpg', 'tif')

        sr = np.array(sr.squeeze(0).permute(1, 2, 0).data.cpu())
        sr = sr[:, :, [2, 1, 0]]

        sr = np.array(sr).astype(np.uint8)

        cv2.imwrite(save_dir + '/' + img_name , sr)

if __name__ == '__main__':
    with torch.no_grad():
        main()
