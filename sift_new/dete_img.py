# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 下午12:14
# @Author  : Shql
# @Site    : 
# @File    : dete_img.py
# @Software: PyCharm
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.cluster.vq import *
from sklearn.externals import joblib
from sklearn import preprocessing
import faiss
import pandas as pd
import time


def deal_sift_voc(file_name):
    print 'begin'
    print time.localtime(time.time())
    numWords = 4000
    path_exp = os.path.expanduser(file_name)
    path_list = os.listdir(path_exp)
    # image_paths = []
    label_num_list = set()
    sift_list = []
    count = 1
    for item in path_list:
        sift = os.path.join(path_exp, item)
        # image_paths.append(sift)
        print 'loading file %s dete num: %d' % (sift, count)
        r = np.load(sift)
        result = r['img']
        labels = r['label']
        label_num_list = label_num_list | set(labels)
        if len(sift_list) != 0:
            if type(result) != None:
                try:
                    sift_list = np.concatenate((sift_list, result), axis=0)
                except:
                    pass
        else:
            sift_list = result
        count += 1

    label_num_list = list(label_num_list)
    # sift_list_set = np.array(list(set([tuple(t) for t in sift_list])))
    sift_list_set = np.array(list(sift_list))
    print 'all lens is :%d' % len(sift_list_set)
    # sift_list = np.array(sift_list)
    sift_list_set = np.float32(sift_list_set)
    print 'kmeans sift file......'
    kmeans = faiss.Kmeans(128, numWords, niter=10, verbose=True)
    kmeans.train(sift_list_set)
    voc = kmeans.centroids
    # voc, variance = kmeans(sift_list, numWords, 1)
    print 'kmeans over'
    print time.localtime(time.time())

    # Calculate the histogram of features

    im_features = np.zeros((len(label_num_list), numWords), "float32")
    indexs = faiss.IndexFlatL2(128)
    print(indexs.is_trained)
    indexs.add(voc)

    for pathss in path_list:
        print 'dete file :%s' % pathss
        sift_yuan = os.path.join(path_exp, pathss)
        r = np.load(sift_yuan)
        lable_np = r['label']
        sift_np = r['img']
        # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label

        # words, distance = vq(sift_np, voc)
        distance, words = indexs.search(sift_np, k=1)
        for i in range(0, len(words)):
            index = label_num_list.index(lable_np[i])
            im_features[index][words[i][0]] += 1

    print 'calculate tfidf....'
    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(im_features) * 1000 + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Perform L2 normalization
    im_features = im_features * idf
    im_features = preprocessing.normalize(im_features, norm='l2')

    print 'calculate over ....'
    print time.localtime(time.time())

    print 'saving model'

    joblib.dump((im_features, label_num_list, idf, numWords, voc), "bofvoc1.pkl", compress=3)


# def dete_sift_256OC(file_name):
#     print 'begin'
#     print time.localtime(time.time())
#     numWords = 4000
#     path_exp = os.path.expanduser(file_name)
#     path_list = os.listdir(path_exp)
#     label_num_list = set()
#     sift_list = []
#     count = 1
#     for item in path_list:
#         sift = os.path.join(path_exp, item)
#         print 'loading file %s dete num: %d' % (sift, count)
#         r = np.load(sift)
#         result = r['img']
#         labels = r['labels']
#         label_num_list = label_num_list | set(labels)
#         if len(sift_list) != 0:
#             if type(result) != None:
#                 try:
#                     sift_list = np.concatenate((sift_list, result), axis=0)
#                 except:
#                     pass
#         else:
#             sift_list = result
#         count += 1
#
#     label_num_list = list(label_num_list)
#     # sift_list_set = np.array(list(set([tuple(t) for t in sift_list])))
#     sift_list_set = np.array(list(sift_list))
#     print 'all lens is :%d' % len(sift_list_set)
#     # sift_list = np.array(sift_list)
#     sift_list_set = np.float32(sift_list_set)
#     print 'kmeans sift file......'
#     kmeans = faiss.Kmeans(128, numWords, niter=10, verbose=True)
#     kmeans.train(sift_list_set)
#     voc = kmeans.centroids
#     # voc, variance = kmeans(sift_list, numWords, 1)
#     print 'kmeans over'
#     print time.localtime(time.time())
#
#     # Calculate the histogram of features
#
#     im_features = np.zeros((len(label_num_list), numWords), "float32")
#     indexs = faiss.IndexFlatL2(128)
#     print(indexs.is_trained)
#     indexs.add(voc)
#
#     for pathss in path_list:
#         print 'dete file :%s' % pathss
#         sift_yuan = os.path.join(path_exp, pathss)
#         r = np.load(sift_yuan)
#         lable_np = r['labels']
#         sift_np = r['img']
#         # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
#
#         # words, distance = vq(sift_np, voc)
#         distance, words = indexs.search(sift_np, k=1)
#         for i in range(0, len(words)):
#             index = label_num_list.index(lable_np[i])
#             im_features[index][words[i][0]] += 1
#
#     print 'calculate tfidf....'
#     # Perform Tf-Idf vectorization
#     nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
#     idf = np.array(np.log((1.0 * len(im_features) * 1000 + 1) / (1.0 * nbr_occurences + 1)), 'float32')
#
#     # Perform L2 normalization
#     im_features = im_features * idf
#     im_features = preprocessing.normalize(im_features, norm='l2')
#
#     print 'calculate over ....'
#     print time.localtime(time.time())
#
#     print 'saving model'
#
#     joblib.dump((im_features, label_num_list, idf, numWords, voc), "bof6000.pkl", compress=3)


def read_sift():
    # target = '/Users/shi/work/data/tain/fearture/'
    # target = '/Users/shi/work/data/tain/zhenghe/'
    target = '/Users/shi/work/data/cate_img/'
    deal_sift_voc(target)


if __name__ == '__main__':
    # write_sift()
    read_sift()
