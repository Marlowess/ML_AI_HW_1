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
X_t = [] # here I save the eigenvectors of my dataset according to the number of PC

# This method opens each class folder and gets raw pixels of each image
def getData(directory_name, x, label, y):
    directory = os.fsencode(directory_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        i = Image.open(directory_name + filename)#.convert('L')
        x.extend(np.asarray(i))#.ravel())
        global count
        y.insert(count, label)
        count += 1
        global numbers
        numbers[label]  = numbers[label] + 1

# This methos applies the PCA to the data with number of components as decided by user
def pcaApplication(x_r, number_of_components, scaler, image_index = 99): #image_index = 99
    my_pca = PCA(number_of_components)
    X_t = my_pca.fit_transform(x_r)
    imgs_compressed = my_pca.inverse_transform(X_t)
    test_image = imgs_compressed[image_index]
    test_image = scaler.inverse_transform(test_image)
    test_image = np.reshape(test_image, (227,227,3))
    variance = my_pca.explained_variance_ratio_.cumsum()[number_of_components-1]
    return test_image, variance

# Plot one or more images
def plotImage(test_img, variance, number_of_components):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    img = Image.fromarray(test_img.astype('uint8'))
    imgplot = plt.imshow(img)
    a.set_title(str(number_of_components) + ' Principal Components\nVariance: ' + str(variance))
    plt.savefig(rootFolder + 'my_pca_' + str(number_of_components) + '.jpg')

# 1.2 Principal Components Visualization
# Loading images dataset
rootFolder = '/home/stefano/Documenti/Politecnico/Magistrale/2 Anno/ML/Homework/#1/PACS_homework/' # root images folder
folder1 = 'dog/'
folder2 = 'guitar/'
folder3 = 'house/'
folder4 = 'person/'

getData(rootFolder+folder1, x, 0, y) # subset of dog images
getData(rootFolder+folder2, x, 1, y) # subset of guitar images
getData(rootFolder+folder3, x, 2, y) # subset of house images
getData(rootFolder+folder4, x, 3, y) # subset of person images


# Computing PCA on the matrix
x = np.asarray(x, dtype=np.float64) # all 3D images
x_r = np.reshape(x, (1087,154587)) # vectorial representation of matrix
#x_r = np.reshape(x, (189,154587)) # vectorial representation of matrix
scaler = StandardScaler()
x_r = scaler.fit_transform(x_r)

# my_compressed_image_60, variance_60 = pcaApplication(x_r, 60, scaler)
# plotImage(my_compressed_image_60, variance_60, 60)
# my_compressed_image_6, variance_6 = pcaApplication(x_r, 6, scaler)
# plotImage(my_compressed_image_6, variance_6, 6)
my_compressed_image_2, variance_2 = pcaApplication(x_r, 2, scaler)
plotImage(my_compressed_image_2, variance_2, 2)


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
