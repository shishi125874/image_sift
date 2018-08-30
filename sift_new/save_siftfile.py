# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 下午3:51
# @Author  : Shql
# @Site    : 
# @File    : save_siftfile.py
# @Software: PyCharm
import os
import numpy as np
import time
import cv2


def sift_dete(file_name):
    path_exp = os.path.expanduser(file_name)
    path_list = os.listdir(path_exp)
    # image_paths = []
    count = 0
    iter = 0
    sift_list = []
    img_name = []
    start = time.clock()
    for item in path_list:
        sift = os.path.join(path_exp, item)
        # image_paths.append(sift)
        print 'iter:%d  loading file %s dete num: %d' % (iter, sift, count)
        r = np.load(sift)
        result = r['img']
        if len(sift_list) != 0:
            if type(result) != None:
                try:
                    label = [item for i in range(0, len(result))]
                    img_name = np.concatenate((img_name, label), axis=0)
                    sift_list = np.concatenate((sift_list, result), axis=0)
                except:
                    pass
        else:
            sift_list = result
            img_name = [item for i in range(0, len(result))]
        count += 1
        if count > 1000:
            print 'saving np data'
            # sift_list = np.array(list(set([tuple(t) for t in sift_list])))
            print len(sift_list), len(img_name)
            np.savez('/Users/shi/work/data/tain/zhenghe/siftEm' + str(iter), img=sift_list, labels=img_name)
            end = time.clock()
            print 'use time:'
            print end - start
            start = time.clock()
            count = 0
            iter += 1
            sift_list = []
            img_name = []


def save_batch_dete(file_name, output_path):
    sift = cv2.xfeatures2d.SIFT_create()
    path_exp = os.path.expanduser(file_name)
    path_list = os.listdir(path_exp)
    count = 0
    sift_list = []
    labels_sift = []
    iter = 0
    for it in path_list:
        folder_path = os.path.join(path_exp, it)
        folder_list = os.listdir(folder_path)
        for item in folder_list:
            file_path = os.path.join(folder_path, item)
            img_name = os.path.splitext(os.path.basename(item))[0]
            img = cv2.imread(file_path)
            # print img.shape[2]
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                kp, des = sift.detectAndCompute(gray, None)
                des = np.array(des)
                label = [it + '/' + img_name for i in range(0, len(des))]
            except:
                continue

            print 'iter:%d  loading file %s , sift lenght: %d dete num: %d' % (iter, file_path, len(des), count)
            if count == 0:
                sift_list = des
                labels_sift = label
            elif count < 1000:
                try:
                    sift_list = np.concatenate((sift_list, des), axis=0)
                    labels_sift = np.concatenate((labels_sift, label), axis=0)
                    # print 'all sift batch lenght is:' + str(sift_list.shape)
                except:
                    pass
            else:
                try:
                    sift_list = np.concatenate((sift_list, des), axis=0)
                    labels_sift = np.concatenate((labels_sift, label), axis=0)
                except:
                    pass
                print 'all sift batch lenght is:%d' % len(sift_list)
                np.savez(output_path + 'siftbatch' + str(iter), img=sift_list, label=labels_sift)
                sift_list = []
                labels_sift = []
                count = 0
                iter += 1
            count += 1


if __name__ == '__main__':
    # sift_dete('/Users/shi/work/data/tain/fearture/')
    save_batch_dete('/Users/shi/Downloads/256_ObjectCategories', '/Users/shi/work/data/cate_image_feature/')
