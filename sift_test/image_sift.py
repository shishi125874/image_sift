# -*- coding: utf-8 -*-
# @Time    : 2018/7/30 下午1:58
# @Author  : Big brother Li
# @Site    : 
# @File    : image_sift.py
# @Software: PyCharm

import cv2
import faiss
import os
import numpy as np


def duplicate_removal(xy):
    if xy.shape[0] < 2:
        return xy
    _tmp = (xy * 4000).astype('i4')  # 转换成 i4 处理
    _tmp = _tmp[:, 0] + _tmp[:, 1] * 1j  # 转换成复数处理
    keep = np.unique(_tmp, return_index=True)[1]  # 去重 得到索引
    return xy[keep]  # 得到数据并返回


# _tmp[:,0] 切片操作，因为时二维数组，_tmp[a:b, c:d]为通用表达式，
# 表示取第一维的索引 a 到索引 b，和第二维的索引 c 到索引 d
# 当取所有时可以直接省略，但要加':'冒号 、当 a == b 时可只写 a ,同时不用':'冒号

def read_file(file_name):
    path_exp = os.path.expanduser(file_name)
    path_list = os.listdir(path_exp)
    img_list = []
    for item in path_list:
        img = os.path.join(path_exp, item)
        img_list.append(img)

    # print img_list
    return img_list


def sift_dete(img_list):
    sift = cv2.xfeatures2d.SIFT_create()
    des_all = []
    count = 1
    label = []
    iter = 0
    for path in img_list:
        for i in range(0, int(len(img_list) / 3000)):
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            des = np.array(des)
            print('%d dete image %s , iter:%d, count : %d ,lens : %d ' % (
                i, path, iter, count, len(des_all)))
            if len(des_all) == 0:
                des_all = des
            else:
                if len(kp) != 0:
                    des_all = np.concatenate((des_all, des), axis=0)

            if iter == 500:
                des_all = np.array(list(set([tuple(t) for t in des_all])))
                iter = 0

            label.append(path)
            count += 1
            iter += 1
        np.savez('siftEm1', img=des_all, label=label)

    np.savez('siftEm', img=des_all, label=label)


def kmeans_sift():
    folder_path = '/Users/shi/work/diaoyan/SSD-Tensorflow/embbings.npz'
    # folder_path = '/Users/shi/work/diaoyan/dali_face/embbings.npz'
    f = open(folder_path)
    r = np.load(folder_path)
    embbiding = r['img']

    data = np.asarray(embbiding, dtype=np.float32)

    kmeans = faiss.Kmeans()

    d = 128


if __name__ == '__main__':
    imgpath = '/Users/shi/Downloads/VOCdevkit/VOC2012/JPEGImages/'
    img_list = read_file(imgpath)
    sift_dete(img_list)
