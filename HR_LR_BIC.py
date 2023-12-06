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
                        default='/home/damon/Downloads/DASR-main/remote_sensing/Data/WHU-RS19/VAL',
                        help='image directory')
    parser.add_argument('--scale', type=str, default='4',
                        help='super resolution scale')
    parser.add_argument('--resume', type=int, default=368,
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
    save_dir = dir + '/results/BICUBIC/RSSCN7/gParking'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    eval_psnr = 0
    eval_ssim = 0
    eval_lpips = 0

    string = args.img_dir + '/*.jpg'
    coll = io.ImageCollection(string)

    for i in range(len(coll)):

        ###
        hr = pil_image.open(coll.files[i]).convert('RGB')
        hr_width = (hr.width // int(args.scale[0])) * int(args.scale[0])
        hr_height = (hr.height // int(args.scale[0])) * int(args.scale[0])
        lr = hr.resize(((hr_width // int(args.scale[0])), hr_height // int(args.scale[0])), resample=pil_image.BICUBIC)
        sr_b = lr.resize(((lr.width * int(args.scale[0])), lr.height * int(args.scale[0])), resample=pil_image.BICUBIC)
        hr = hr.resize(((lr.width * int(args.scale[0])), lr.height * int(args.scale[0])), resample=pil_image.BICUBIC)
        sr_b = np.array(sr_b).astype(np.uint8)
        lr = np.array(lr).astype(np.uint8)
        lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
        lr = torch.from_numpy(lr).float().cuda().unsqueeze(0).unsqueeze(0)

        # inference
        sr = DASR(lr[:, 0, ...])
        sr = utility.quantize(sr, 255.0)
        ###
        # sr_ = imageio.imread(
        #     '/home/damon/Downloads/DASR-main/experiment/blindsr_x4_bicubic_iso/results/storagetanks03_CBM_x4.0_SR.bmp')
        # psnr = PSNR(sr_, sr)

        # save sr results
        img_name = coll.files[i].replace(args.img_dir + '/', '')
        sr = np.array(sr.squeeze(0).permute(1, 2, 0).data.cpu())
        sr = sr[:, :, [2, 1, 0]]

        cv2.imwrite(save_dir + '/' + img_name , sr)

        #psnr_test
        b, g, r = cv2.split(sr)
        sr = cv2.merge([r, g, b])
        sr = np.array(sr).astype(np.uint8)
        hr = np.array(hr).astype(np.uint8)
        # ###
        # sr_block ,hr_block= np.zeros((35,35,3)).astype(np.uint8) ,np.zeros((35,35,3)).astype(np.uint8)
        # sr_block = sr_b[0:35, 256 - 35:256]
        # hr_block = hr[0:35, 256 - 35:256]
        # sr_block = np.ascontiguousarray(sr_block.transpose((2, 0, 1)))
        # sr_block = torch.from_numpy(sr_block).float().cuda().unsqueeze(0)
        # hr_block_ = np.ascontiguousarray(hr_block.transpose((2, 0, 1)))
        # hr_block_ = torch.from_numpy(hr_block).float().cuda().unsqueeze(0)
        psnr = calculate_psnr(sr, hr, 2)
        ssim = calculate_ssim(sr, hr, 2)
        eval_psnr += psnr
        eval_ssim += ssim
        dummy_im0 = lpips.im2tensor(hr)
        dummy_im1 = lpips.im2tensor(sr)
        dist = loss_fn_alex.forward(dummy_im0, dummy_im1)
        lpip = dist.mean().item()
        eval_lpips += lpip
        print(coll.files[i].replace(args.img_dir + '/', '')+'\t'+'psnr:%.3f ,ssim:%.3f ,lpips:%.3f' % (psnr, ssim, lpip))
        # with open("myModel_RSN_x2.txt", "a") as f:
        #     f.write(coll.files[i].replace(args.img_dir + '/', '') + '\t' +'psnr: '+str(psnr)+'ssim: '+str(ssim)+'lpips: '+str(lpip)+'\t\n')

    print('All: psnr:%.3f ,ssim:%.3f ,lpips:%.3f' % (eval_psnr/len(coll), eval_ssim/len(coll), eval_lpips/len(coll)))
    # with open("myModel_RSN_x2.txt", "a") as f:
    #     f.write('ALL: ' + '\t' + 'psnr: ' + str(eval_psnr/len(coll)) + 'ssim: ' + str(eval_ssim/len(coll)) + 'lpips: ' + str(eval_lpips/len(coll)) + '\n\n')

if __name__ == '__main__':
    with torch.no_grad():
        main()
