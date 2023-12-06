from model.blindsr import BlindSR
# from model.mymodel import BlindSR
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
import time
import lpips
loss_fn_alex = lpips.LPIPS(net='alex')
from ptflops import get_model_complexity_info
from thop import profile
# skimage.measure.compare_psnr(sr_block, hr_block, data_range=255)
# skimage.measure.compare_ssim(sr_block, hr_block, data_range=255 ,multichannel=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str,
                        default='/home/damon/Downloads/DASR-main/remote_sensing/Data/A',
                        help='image directory')
    parser.add_argument('--scale', type=str, default='4',
                        help='super resolution scale')
    parser.add_argument('--resume', type=int, default=793,
                        help='resume from specific checkpoint')
    parser.add_argument('--blur_type', type=str, default='iso_gaussian',
                        help='blur types (iso_gaussian | aniso_gaussian)')
    return parser.parse_args()

def print_network(net):
    # sum(map(lambda x: x.numel(), net.parameters()))
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)
def main():
    args = parse_args()
    if args.blur_type == 'iso_gaussian':
        dir = '/home/damon/Downloads/DASR-main/experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_iso'
    elif args.blur_type == 'aniso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_aniso'

    # path to save sr images
    # save_dir = dir + '/results/WHU-RS19/0-SR'
    save_dir = '/home/damon/Downloads/DASR-main/remote_sensing/tmp'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # model_AID_MSR_E3726.pt
    DASR = BlindSR(args).cuda()
    print_network(DASR)
    # DASR.load_state_dict(torch.load(dir + '/model/model_add_X8_' + str(args.resume) + '.pt'), strict=False)
    DASR.load_state_dict(torch.load('/home/damon/Downloads/DASR-main/experiment/blindsr_x4_bicubic_iso/model/model_add_thelast.pt'), strict=False)
    DASR.eval()
    eval_psnr = 0
    eval_ssim = 0
    eval_lpips = 0

    LR_dir = '/home/damon/Downloads/DASR-main/remote_sensing/Data/B'
    string = args.img_dir + '/*.jpg'
    coll = io.ImageCollection(string)
    LR_coll = io.ImageCollection(LR_dir + '/*.tif')

    t2 = 0
    for i in range(len(LR_coll)):
        ###
        # hr = pil_image.open(coll.files[i]).convert('RGB')
        # hr_width = (hr.width // int(args.scale[0])) * int(args.scale[0])
        # hr_height = (hr.height // int(args.scale[0])) * int(args.scale[0])
        # lr = hr.resize(((hr_width // int(args.scale[0])), hr_height // int(args.scale[0])), resample=pil_image.BICUBIC)
        # sr_b = lr.resize(((lr.width * int(args.scale[0])), lr.height * int(args.scale[0])), resample=pil_image.BICUBIC)
        # hr = hr.resize(((lr.width * int(args.scale[0])), lr.height * int(args.scale[0])), resample=pil_image.BICUBIC)
        # sr_b = np.array(sr_b).astype(np.uint8)

        lr = pil_image.open(LR_coll.files[i]).convert('RGB')
        lr = np.array(lr).astype(np.uint8)
        lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
        lr = torch.from_numpy(lr).float().cuda().unsqueeze(0).unsqueeze(0)

        # ##
        # lr = imageio.imread(args.img_dir)
        # lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
        # lr = torch.from_numpy(lr).float().cuda().unsqueeze(0).unsqueeze(0)
        t0 = time.time()
        # inference
        sr = DASR(lr[:, 0, ...])
        sr = utility.quantize(sr, 255.0)
        ###
        # sr_ = imageio.imread(
        #     '/home/damon/Downloads/DASR-main/experiment/blindsr_x4_bicubic_iso/results/storagetanks03_CBM_x4.0_SR.bmp')
        # psnr = PSNR(sr_, sr)

        # save sr results
        img_name = LR_coll.files[i].replace(LR_dir + '/', '')
        # img_name = img_name.replace('low', 'answer')
        # img_name = img_name.replace('jpg', 'tif')

        sr = np.array(sr.squeeze(0).permute(1, 2, 0).data.cpu())
        sr = sr[:, :, [2, 1, 0]]

        sr = np.array(sr).astype(np.uint8)
        t1 = time.time()
        t2 = t2 +(t1-t0)

        cv2.imwrite(save_dir + '/' + img_name , sr)

    print("======>ALL Timer: %.6f sec." % (t2 / 56))
    # flops, params = get_model_complexity_info(DASR, (3, 150, 150), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)
        #psnr_test
        # b, g, r = cv2.split(sr)
        # sr = cv2.merge([r, g, b])
        # sr = np.array(sr).astype(np.uint8)
        # hr = np.array(hr).astype(np.uint8)
        # ###
        # sr_block ,hr_block= np.zeros((35,35,3)).astype(np.uint8) ,np.zeros((35,35,3)).astype(np.uint8)
        # sr_block = sr_b[0:35, 256 - 35:256]
        # hr_block = hr[0:35, 256 - 35:256]
        # sr_block = np.ascontiguousarray(sr_block.transpose((2, 0, 1)))
        # sr_block = torch.from_numpy(sr_block).float().cuda().unsqueeze(0)
        # hr_block_ = np.ascontiguousarray(hr_block.transpose((2, 0, 1)))
        # hr_block_ = torch.from_numpy(hr_block).float().cuda().unsqueeze(0)
    #     psnr = calculate_psnr(sr, hr, 2)
    #     ssim = calculate_ssim(sr, hr, 2)
    #     eval_psnr += psnr
    #     eval_ssim += ssim
    #     dummy_im0 = lpips.im2tensor(hr)
    #     dummy_im1 = lpips.im2tensor(sr)
    #     dist = loss_fn_alex.forward(dummy_im0, dummy_im1)
    #     lpip = dist.mean().item()
    #     eval_lpips += lpip
    #     print(coll.files[i].replace(args.img_dir + '/', '')+'\t'+'psnr:%.3f ,ssim:%.3f ,lpips:%.3f' % (psnr, ssim, lpip))
    #     # with open("myModel_RSN_x2.txt", "a") as f:
    #     #     f.write(coll.files[i].replace(args.img_dir + '/', '') + '\t' +'psnr: '+str(psnr)+'ssim: '+str(ssim)+'lpips: '+str(lpip)+'\t\n')
    #
    # print('All: psnr:%.3f ,ssim:%.3f ,lpips:%.3f' % (eval_psnr/len(coll), eval_ssim/len(coll), eval_lpips/len(coll)))
    # with open("myModel_RSN_x2.txt", "a") as f:
    #     f.write('ALL: ' + '\t' + 'psnr: ' + str(eval_psnr/len(coll)) + 'ssim: ' + str(eval_ssim/len(coll)) + 'lpips: ' + str(eval_lpips/len(coll)) + '\n\n')

if __name__ == '__main__':
    with torch.no_grad():
        main()




    # lr = pil_image.open(LR_coll.files[0]).convert('RGB')
    # cropped_2 = lr.crop((lr.size[0] / 5 * 1, 0, lr.size[0] / 5 * 2, 1733))
    # cropped_2.show()
    # cropped_1 = lr.crop((0, 0, lr.size[0] / 5, 1733))
    # cropped_1.show()
    # cropped_3 = lr.crop((lr.size[0] / 5 * 2, 0, lr.size[0] / 5 * 3, 1733))
    # cropped_3.show()
    # cropped_4 = lr.crop((lr.size[0] / 5 * 3, 0, lr.size[0] / 5 * 4, 1733))
    # cropped_4.show()
    # cropped_5 = lr.crop((lr.size[0] / 5 * 4, 0, lr.size[0] / 5 * 5, 1733))
    # cropped_5.show()
    # cropped_1 = lr.crop((0, 0, 1549, 1733))
    # cropped_1.show()
    # cropped_1.save("/home/damon/Downloads/DASR-main/remote_sensing/tmp/cut/cropped_1.tif")
    # cropped_2.save("/home/damon/Downloads/DASR-main/remote_sensing/tmp/cut/cropped_2.tif")
    # cropped_3.save("/home/damon/Downloads/DASR-main/remote_sensing/tmp/cut/cropped_3.png")
    # cropped_4.save("/home/damon/Downloads/DASR-main/remote_sensing/tmp/cut/cropped_4.png")
    # cropped_5.save("/home/damon/Downloads/DASR-main/remote_sensing/tmp/cut/cropped_5.png")
    # joint = pil_image.new('RGB', (cropped_1.size[0] + cropped_2.size[0], cropped_1.size[1]))
    # loc1, loc2 = (0, 0), (cropped_1.size[0], 0)
    # joint.paste(cropped_1, loc1)
    # joint.paste(cropped_2, loc2)
    # joint.show()
    # LR_dir = '/home/damon/Downloads/DASR-main/remote_sensing/tmp'
    # LR_coll = io.ImageCollection(LR_dir + '/*.png')
    # cropped_1 = pil_image.open(LR_coll.files[0]).convert('RGB')
    # cropped_2 = pil_image.open(LR_coll.files[1]).convert('RGB')
    # cropped_3 = pil_image.open(LR_coll.files[2]).convert('RGB')
    # cropped_4 = pil_image.open(LR_coll.files[3]).convert('RGB')
    # cropped_5 = pil_image.open(LR_coll.files[4]).convert('RGB')
    # joint = pil_image.new('RGB', (
    # cropped_1.size[0] + cropped_2.size[0] + cropped_3.size[0] + cropped_4.size[0] + cropped_5.size[0],
    # cropped_1.size[1]))
    # loc1, loc2, loc3, loc4, loc5 = (0, 0), (cropped_1.size[0], 0), (cropped_1.size[0] + cropped_2.size[0], 0), (
    # cropped_1.size[0] + cropped_2.size[0] + cropped_3.size[0], 0), (
    #                                cropped_1.size[0] + cropped_2.size[0] + cropped_3.size[0] + cropped_4.size[0], 0)
    # joint.paste(cropped_1, loc1)
    # joint.paste(cropped_2, loc2)
    # joint.paste(cropped_3, loc3)