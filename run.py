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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import copy

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


# This method gets the last 6 principal components from the matrix
# sklearn libraries don't provide a method to do it, so I've to perform the operations
# manually on data
# index1 = index of initial principal components
# index2 = index of final principal components + 1
def getPCResults(X_t, my_pca, index1, index2):
    index1, index2 = int(index1), int(index2)
    my_eig = my_pca.components_[index1:index2]
    remain = index2 - index1
    my_pca.components_[0:remain] = my_eig[0:remain]
    my_pca.components_[remain:] = 0
    imgs_compressed = my_pca.inverse_transform(X_t)
    variance = my_pca.explained_variance_ratio_.cumsum()[remain - 1]
    return imgs_compressed, variance

# Just the computation of all principal components
def getAllPC(x_r):
    my_pca = PCA()
    X_t = my_pca.fit_transform(x_r)
    return X_t, my_pca

def getVarianceArray(x_r):
    my_pca = PCA()
    X_t = my_pca.fit_transform(x_r)
    variance = my_pca.explained_variance_ratio_.cumsum()
    # np.set_printoptions(threshold=np.nan)
    # print(variance)
    # plotSplineFunction(variance)
    return variance

# Gets the chosen image reprojected
def getReprojectedImage(imgs_compressed, scaler, image_index=99):
    test_image = imgs_compressed[image_index]
    test_image = scaler.inverse_transform(test_image)
    test_image = np.reshape(test_image, (227,227,3))
    return test_image

# Plots one or more images
def plotImage(test_img, variance, number_of_components):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    img = Image.fromarray(test_img.astype('uint8'))
    imgplot = plt.imshow(img)
    a.set_title(str(number_of_components) + ' Principal Components\nVariance: ' + str(variance))
    plt.savefig(rootFolder + 'my_pca_' + str(number_of_components) + '.jpg')

# Plots a scatter diagram to visualize principal components
def plotScatter(matrix, components_details, saving_name, index1, index2):
    dogIndex = numbers[0]-1
    guitarIndex = numbers[0]+numbers[1]-1
    houseIndex = numbers[0]+numbers[1]+numbers[2]-1
    personIndex = numbers[0]+numbers[1]+numbers[2]+numbers[3]-1

    colors=["red", "green", "dodgerblue", "yellow"]
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    d = plt.scatter(matrix[0:dogIndex,0], matrix[0:dogIndex,1], marker='o', color=colors[0])
    g = plt.scatter(matrix[dogIndex+1:guitarIndex,0], matrix[dogIndex+1:guitarIndex,1], marker='o', color=colors[1])
    h = plt.scatter(matrix[guitarIndex+1:houseIndex,0], matrix[guitarIndex+1:houseIndex,1], marker='o', color=colors[2])
    p = plt.scatter(matrix[houseIndex+1:personIndex,0], matrix[houseIndex+1:personIndex,1], marker='o', color=colors[3])

    plt.legend((d,g,h,p),
               ('Dog', 'Guitar', 'House', 'Person'),
               loc='lower right',
               ncol=2,
               fontsize=8)
    plt.grid(True)
    a.set_title("Scatter plot of " + components_details + " principal components")
    plt.savefig(rootFolder + saving_name + '.jpg')
    plt.show()

# Plots the variance function related to principal components
def plotSplineFunction(array):
    x_new = np.linspace(0, array.size, array.size)
    y = array
    print(y)
    plt.plot (x_new, y)
    plt.scatter (x_new, y)
    plt.savefig(rootFolder + "variance_plot" + '.jpg')

# Performs a cross-validation on data
def classification(X, Y):
    # First, I've to create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    GaussianNB(priors=None, var_smoothing=1e-09)
    print(clf.score(X_test, y_test))
    return clf

# Performs a prediction according to the classifier and the given image
def image_predictor(image, classifier):
    print(classifier.predict(np.reshape(image, (1,154587))))




# Loads images dataset
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
scaler = StandardScaler()
x_r = scaler.fit_transform(x_r)

X_t, my_pca = getAllPC(x_r)

X_R_60, variance60 = getPCResults(X_t, copy.copy(my_pca), 0, 60)
img_60 = getReprojectedImage(X_R_60, scaler)
plotImage(img_60, variance60, 60)

X_R_6, variance6 = getPCResults(X_t, copy.copy(my_pca), 0, 6)
img_6 = getReprojectedImage(X_R_6, scaler)
plotImage(img_6, variance6, 6)

X_R_2, variance2 = getPCResults(X_t, copy.copy(my_pca), 0, 2)
img_2 = getReprojectedImage(X_R_2, scaler)
plotImage(img_2, variance2, 2)

X_R_l6, variancel6 = getPCResults(X_t, copy.copy(my_pca), 1081, 1087)
img_l6 = getReprojectedImage(X_R_l6, scaler)
plotImage(img_l6, variancel6, 'Last 6')

# getVarianceArray(x_r)
# X_R, variance = getPC(x_r, 0, 1087)
# #img = getReprojectedImage(X_R, scaler)
# print("Variance is {}".format(variance))
# #plotImage(img, variance, 'two')
# X_R = scaler.inverse_transform(X_R)
# #plotScatter(X_R, "tenth and eleventh", "scatter_10_11")


# CLASSIFICATION STEP
# img = np.reshape(x_r[653].astype('uint8'), (227,227,3))
# imgplot = plt.imshow(img)
# classification(X_R, y)
