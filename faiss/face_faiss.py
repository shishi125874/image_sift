# -*- coding: utf-8 -*-
# @Time    : 2018/7/22 下午11:32
# @Author  : Shql
# @Site    : 
# @File    : face_faiss.py
# @Software: PyCharm
import faiss
import numpy as np
import os
import cv2
import time


def show_in_one(images, show_size=(300, 300), blank_size=2, window_name="merge"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("ingnore count %s" % (max_count - count))
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, merge_img)


d = 128

folder_path = '/Users/shi/work/diaoyan/dali_face/embbings.npz'
f = open(folder_path)

r = np.load(folder_path)

face_name = r['labels']
embbiding = r['image']
data = []
face_data = []
for faces in face_name:
    for face in faces:
        face_data.append(face)
for emb in embbiding:
    for em in emb:
        data.append(em)

print len(face_data)

data = np.asarray(data)
print data

# 进行简单的k-近邻搜索
index = faiss.IndexFlatL2(d)  # build the index
print(index.is_trained)
start_suoyin = time.clock()
index.add(data)  # add vectors to the index
end_suoyin = time.clock()
print 'the time in add kmeans:'
print end_suoyin - start_suoyin

print(index.ntotal)
serarch_i = np.asarray([data[700], data[100]])

k = 4  # we want to see 4 nearest neighbors
start = time.clock()
D, I = index.search(serarch_i[:5], k)  # sanity check
end = time.clock()
print end - start

# ----------------------------------------------------------
# 加快搜索
nlist = 100  # 聚类中心的个数
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# here we specify METRIC_L2, by default it performs inner-product search
assert not index.is_trained
index.train(data)
assert index.is_trained

start_k_suoyin = time.clock()
index.add(data)  # add may be a bit slower as well
end_k_suoyin = time.clock()
print 'the time in add kmeans:'
print end_k_suoyin - start_k_suoyin

serarch_a = np.asarray([data[700], data[200]])
# print serarch_a

start = time.clock()
D, I = index.search(serarch_a, k)  # actual search
end = time.clock()

serarch_a1 = np.asarray([data[710], data[201]])
D1, I1 = index.search(serarch_a1, k)

print end - start
result = I
print result  # neighbors of the 5 last queries
print D[-5:]
for i in result:
    print i[0]

# for iten in result[0]:
#     # img = cv2.imread(face_data[iten])
#     # cv2.imshow(face_data[iten], img)
#     # images.append(img)
#     print face_data[iten]

# show_in_one(images)

print '-----------------'

# cv2.waitKey(0)
# cv2.destroyAllWindows()

index.nprobe = 10  # default nprobe is 1, try a few more

serarch_b = np.asarray([data[500]])
D, I = index.search(serarch_b, k)
print(I[-5:])  # neighbors of the 5 last queries
