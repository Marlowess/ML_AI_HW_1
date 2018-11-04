# AI & ML Homework - PoliTo AA 2018/2018
# Prof. Barbara Caputo
# Homework #1 - Stefano Brilli s249914

from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colo
from sklearn.preprocessing import StandardScaler
import pandas as pd

# vector representing how many elements there are in each folder
# 0:dog 1:guitar 2:house 3:person
numbers = [0, 0, 0, 0]
x = [] # list of items
y = []
count = 0

# This method opens each class folder and gets raw pixels of each image
def getData(directory_name, label):
    directory = os.fsencode(directory_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        i = Image.open(directory_name + filename)#.convert('L')
        global x
        x.extend(np.asarray(i))#.ravel())
        global y
        global count
        y.insert(count, label)
        count += 1
        global numbers
        numbers[label]  = numbers[label] + 1



# 1.2 Principal Components Visualization
# Loading images dataset
rootFolder = '/home/stefano/Documenti/Politecnico/Magistrale/2 Anno/ML/Homework/#1/PACS_homework/' # root images folder
folder1 = 'dog/'
folder2 = 'guitar/'
folder3 = 'house/'
folder4 = 'person/'

getData(rootFolder+folder1, 0) # subset of dog images
# getData(rootFolder+folder2, x, 1) # subset of guitar images
# getData(rootFolder+folder3, x, 2) # subset of house images
# getData(rootFolder+folder4, x, 3) # subset of person images


# Computing PCA on the matrix
x = np.asarray(x, dtype=np.float64) # all 3D images
#x_r = np.reshape(x, (1087,154587)) # vectorial representation of matrix
x_r = np.reshape(x, (189,154587)) # vectorial representation of matrix
scal = StandardScaler()
x_r = scal.fit_transform(x_r)

pca60 = PCA(189)
#pca6 = PCA(6)
# pca2 = PCA(2)

X_t = pca60.fit_transform(x_r) # dataset drawed according to its sixty first principal components
#X_t = pca6.fit_transform(x_r) # dataset drawed according to its six first principal components
# X_t = pca2.fit_transform(x_r) # dataset drawed according to its two first principal components

# Applying pca on images and visualizing the result
imgs_compressed = pca60.inverse_transform(X_t)
test_image = imgs_compressed[99]
test_image = scal.inverse_transform(test_image)
test_image = np.reshape(test_image, (227,227,3))

# Compressed image visualization
Image.fromarray(test_image.astype('uint8')).show()
print(pca60.explained_variance_ratio_.cumsum())


# Plotting data
dogIndex = numbers[0]-1
guitarIndex = numbers[0]+numbers[1]-1
houseIndex = numbers[0]+numbers[1]+numbers[2]-1
personIndex = numbers[0]+numbers[1]+numbers[2]+numbers[3]-1

colors=["red", "green", "dodgerblue", "black"]

d = plt.scatter(X_t[0:dogIndex,0], X_t[0:dogIndex,1], marker='o', color=colors[0])
g = plt.scatter(X_t[dogIndex+1:guitarIndex,0], X_t[dogIndex+1:guitarIndex,1], marker='o', color=colors[1])
h = plt.scatter(X_t[guitarIndex+1:houseIndex,0], X_t[guitarIndex+1:houseIndex,1], marker='o', color=colors[2])
p = plt.scatter(X_t[houseIndex+1:personIndex,0], X_t[houseIndex+1:personIndex,1], marker='o', color=colors[3])

plt.legend((d,g,h,p),
           ('Dog', 'Guitar', 'House', 'Person'),
           loc='upper right',
           ncol=2,
           fontsize=8)
plt.grid(True)
plt.show()
