from PIL import Image
import numpy as np
import os

# This method opens each class folder and gets raw pixels of each image
def getData(directory_name, x):
    directory = os.fsencode(directory_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        i = Image.open(directory_name + filename)
        x.append(np.asarray(i).ravel())


x = [] # list of items
rootFolder = 'PACS_homework/' # root images folder
folder1 = 'dog/'
folder2 = 'guitar/'
folder3 = 'house/'
folder4 = 'person/'

getData(rootFolder + folder1, x)
getData(rootFolder + folder2, x)
getData(rootFolder + folder3, x)
getData(rootFolder + folder4, x)
