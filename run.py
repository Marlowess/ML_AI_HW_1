from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colo

numbers = [0, 0, 0, 0]

# This method opens each class folder and gets raw pixels of each image
def getData(directory_name, x, y, count, label):
    directory = os.fsencode(directory_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        i = Image.open(directory_name + filename)
        x.append(np.asarray(i).ravel())
        y.insert(count, label)
        count += 1
        global numbers
        numbers[label]  = numbers[label] + 1

def normalize(matrix):
    matrix = (matrix - np.mean(matrix, axis=0, dtype=np.float64))/(np.std(matrix, axis=0,dtype=np.float64))
    return matrix


x = [] # list of items
y = []
count = 0
rootFolder = '/home/stefano/Documenti/Politecnico/Magistrale/2 Anno/ML/Homework/#1/PACS_homework/' # root images folder
folder1 = 'dog/'
folder2 = 'guitar/'
folder3 = 'house/'
folder4 = 'person/'

getData(rootFolder+folder1, x, y, count, 0)
getData(rootFolder+folder2, x, y, count, 1)
getData(rootFolder+folder3, x, y, count, 2)
getData(rootFolder+folder4, x, y, count, 3)

#print(numbers)


x = np.asarray(x, dtype=np.float64)
x = (x - np.mean(x, axis=0, dtype=np.float64))/(np.std(x, axis=0,dtype=np.float64))
cov_x = np.cov(x)
np.set_printoptions(threshold=np.nan)
#print(cov_x)
#y = np.asarray(y).ravel()

X_t = PCA(2).fit_transform(cov_x)

#plt.scatter(X_t[:,0], X_t[:,1], c=y, cmap=colo.ListedColormap(colors))
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
