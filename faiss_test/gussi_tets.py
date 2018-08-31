# -*- coding: utf-8 -*-
# @Time    : 2018/7/17 上午11:14
# @Author  : Shql
# @Site    : 
# @File    : gussi_tets.py
# @Software: PyCharm

import cv2
import numpy as np


def cos(vector1, vector2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


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

a = data[700]
b = data[600]

print cos(a, b)
