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
y = []
count = 0

# This method opens each class folder and gets raw pixels of each image
def getData(directory_name, x, label):
    directory = os.fsencode(directory_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        i = Image.open(directory_name + filename)#.convert('L')
        x.append(np.asarray(i))#.ravel())
        global y
        global count
        y.insert(count, label)
        count += 1
        global numbers
        numbers[label]  = numbers[label] + 1


# Step 1 - loading dataset
x = [] # list of items
rootFolder = '/home/stefano/Documenti/Politecnico/Magistrale/2 Anno/ML/Homework/#1/PACS_homework/' # root images folder
folder1 = 'dog/'
folder2 = 'guitar/'
folder3 = 'house/'
folder4 = 'person/'

getData(rootFolder+folder1, x, 0)
getData(rootFolder+folder2, x, 1)
getData(rootFolder+folder3, x, 2)
getData(rootFolder+folder4, x, 3)


# Step 2 - PCA on a single image
img = np.asarray(Image.open(rootFolder+folder1+'056_0024.jpg'), dtype=np.float64)
img_r = np.reshape(img, (227, 681))
#plt.imshow(img)
img_r = StandardScaler().fit_transform(img_r)

my_pca60 = PCA(60).fit(img_r)
my_pca6 = PCA(6).fit(img_r)
my_pca2 = PCA(2).fit(img_r)

img_compressed = my_pca60.transform(img_r)
temp = my_pca60.inverse_transform(img_compressed)
temp = np.reshape(temp, (227,227,3))
Image.fromarray(temp.astype('uint8')).show()



# Step 3 - Plotting X_t into a scatter plot
x = np.asarray(x, dtype=np.float64)
x_r = np.reshape(x, (1087,154587))
x_r = StandardScaler().fit_transform(x_r)
pca60 = PCA(2).fit(x_r)
X_t = pca60.transform(x_r)

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
