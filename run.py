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
        i = Image.open(directory_name + filename)
        x.append(np.asarray(i).ravel())
        global y
        global count
        y.insert(count, label)
        count += 1
        global numbers
        numbers[label]  = numbers[label] + 1


# Step 1
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



# Step 2
img = Image.open(rootFolder+folder1+'056_0026.jpg')
im_grey = img.convert('L')
im_array = np.array(im_grey)
#print(im_array.shape)
img_pca = PCA(60).fit_transform(im_array)
print(img_pca.shape)
Image.fromarray(img_pca).show()


# Step 3
x = np.asarray(x, dtype=np.float64)
x = StandardScaler().fit_transform(x)
np.set_printoptions(threshold=np.nan)
X_t = PCA(60).fit_transform(x)

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
