# import numpy as np
# import cv2
# import os
#
# # img_h, img_w = 32, 32
# img_h, img_w = 32, 48  # 根据自己数据集适当调整，影响不大
# means, stdevs = [], []
# img_list = []
#
# imgs_path = '/home/damon/Downloads/DASR-main/remote_sensing/Data/DIV2K/HR'
# imgs_path_list = os.listdir(imgs_path)
#
# len_ = len(imgs_path_list)
# i = 0
# for item in imgs_path_list:
#     img = cv2.imread(os.path.join(imgs_path, item))
#     img = cv2.resize(img, (img_w, img_h))
#     img = img[:, :, :, np.newaxis]
#     img_list.append(img)
#     i += 1
#     print(i, '/', len_)
#
# imgs = np.concatenate(img_list, axis=3)
# imgs = imgs.astype(np.float32) / 255.
#
# for i in range(3):
#     pixels = imgs[:, :, i, :].ravel()  # 拉成一行
#     means.append(np.mean(pixels))
#     stdevs.append(np.std(pixels))
#
# # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
# means.reverse()
# stdevs.reverse()
#
# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))
# # normMean = [0.4481487, 0.4372067, 0.4041981]

import cv2, os, argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/home/damon/Downloads/DASR-main/remote_sensing/Data/WHU-RS19/HR')
    args = parser.parse_args()
    return args

def main():
    opt = parse_args()
    img_filenames = os.listdir(opt.dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(opt.dir + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])

if __name__ == '__main__':
    main()
#DIV2K
# [0.44845608 0.43749626 0.40452776]
#AID
# [0.39779539 0.40924516 0.36850663]
#WHU-RS19
#[0.42598359 0.44791445 0.40230705]
