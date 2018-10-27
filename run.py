from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# This method opens each class folder and gets raw pixels of each image
def getData(directory_name, x, y, count, label):
    directory = os.fsencode(directory_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        i = Image.open(directory_name + filename)
        x.append(np.asarray(i).ravel())
        y.insert(count, label)
        count += 1


x = [] # list of items
y = []
count = 0
rootFolder = '/home/stefano/Documenti/Politecnico/Magistrale/2 Anno/ML/Homework/#1/PACS_homework/' # root images folder
folder1 = 'dog/'
folder2 = 'guitar/'
folder3 = 'house/'
folder4 = 'person/'

getData(rootFolder + folder1, x, y, count, 1)
getData(rootFolder + folder2, x, y, count, 2)
getData(rootFolder + folder3, x, y, count, 3)
getData(rootFolder + folder4, x, y, count, 4)

x = np.asarray(x)

X_t = PCA(2).fit_transform(x)
plt.scatter(X_t[:,0], X_t[:,1], c=y)
