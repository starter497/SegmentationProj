# -*- coding: utf-8 -*-
"""
Created on March 20th

@author: cnem modified Neils original code

"""
import os

#import pandas as pd

import numpy as np

import gudhi  #create complexes

import matplotlib

import  time

from PIL import Image, ImageOps

from matplotlib import pyplot as plt

import persistencecurves as pc # vectorization

import warnings

from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

usetex = matplotlib.checkdep_usetex(True) #I dont have latex)










#---------------------Creating functions -----------------


def fractional_lifespancurve(image, height, width , pixel):
    """ Returns a window (needs height h, weight w)
    based on input dimensions and pixel location

    Args
    --------
    image: numpy array
        image to be sliced casted as numpy array
    
    height: int
        desired height for the window image
        
    width: int
        desired width for the window image
    
    pixel: numpy array
        coordinates of where the window will be created

    Returns
    --------
    padded_window: numpy array
        the window sliced from the original image;
        pads the image with 0's if indexing goes
        beyond the image dimensions
        
    """

    # convert image if not numpy array
    if type(image) is not np.array:
        image = np.array(image)

    h = height
    w = width

    n, m = image.shape
    x,y = pixel
    window = np.zeros([h,w])
    for i in range(x, x+h):
        for j in range(y, y+w):
            if i >= n-1 and j >= m-1:
                window[i-x,j-y] = 0
            else:
                window[i-x,j-y] = image[i,j] #padding out of bounds entry with 0s
    fraction_image = window


    reshaped = np.reshape(fraction_image, [fraction_image.shape[0]*fraction_image.shape[1]], order = 'F')
    # reshapes image to a single vector

    Complex = gudhi.CubicalComplex(dimensions=fraction_image.shape, top_dimensional_cells=reshaped)
    # creates the cubical complex from the image

    Complex.persistence()
    Dgm0=Complex.persistence_intervals_in_dimension(0)
    # compute oth dimensional persistence diagram

    Dgm1=Complex.persistence_intervals_in_dimension(1)
    # computes 1st dimensional persistenence diagram

    return Dgm0,fraction_image

#--------------------Configuration---------------------- 

path = r"/home/team3/Documents/imageSegmentation/BSDS300-images/BSDS300/images/train/"

ID ="310007"

import os

import matplotlib.pyplot as plt



script_dir = os.path.dirname(__file__)

results_dir = os.path.join(script_dir, ID + 'Betticurves/')

if not os.path.isdir(results_dir):

    os.makedirs(results_dir)
#Creates a file directory depending on ID of image (useful for saving images)



im = []
im = Image.open(path + ID +".jpg")
im = ImageOps.grayscale(im)
im = np.array(im)

#creating a grayscale pixel array

print("original size :", im.shape)

#Uncomment below this to perform on all train files

'''
train_filenames = os.listdir(path)

print(len(train_filenames))

im =  []

for j in range(len(train_filenames)):
    im = Image.open(path + train_filenames[j])
    im = ImageOps.grayscale(im)
    im = np.array(im)
    print(j)   
    print(im.shape)

'''

x = 0
y = 0
 #x, y respectively or rows and column entry
  #height, width respectively

n,m = im.shape

h = 20 #height    
w = 20 #width
s = 5 #slide

x_1 = np.linspace(0,255,num=256)
#initializing plot 


start = time.time()


#------------------------ Post Processing/ plotting -------------------------------

'''This will iterate the pixels
to make our windows slide depending on increments
'''


BettiArrays = []

while y+w <= m:


    print("x=:",x)
    print("y=:",y)

    pixel = x,y

    [Dgm0,fraction_image]= fractional_lifespancurve(im, h, w , pixel)
    
    print("This is the window:",fraction_image)
    print("This is the window size:", fraction_image.shape)

    '''#uncomment this if you want to see the window images sliding
    plt.imshow(fraction_image)
    plt.show()
    '''

    D0 = pc.Diagram(Dgm =Dgm0, globalmaxdeath = None, infinitedeath=None, inf_policy="keep")
    dgm0 = D0.Betticurve(meshstart=0,meshstop=256,num_in_mesh=256)

    BettiArrays.append(dgm0) #creates a list of betti arrays


    plt.clf()
    
    '''#uncomment this if you want to see the betticurves for windows
    plt.plot(x_1, dgm0)

    plt.savefig(results_dir + "ID:" + ID +" x_"+str(x) + "y_" + str(y)+".png")
    print("betticurvearray:", dgm0)
    '''
    y = y+s
    if y+w > m :
        y=0
        x = x +s
    if x+h >= n:

        break
     


    
    
    



end = time.time()

print("Time elapsed:", end-start)
print("size of All Betti DATA:",len(BettiArrays))

clustering = KMeans(n_clusters=2,
 random_state=0).fit(BettiArrays)

print("Labels in BettiArrays:",clustering.labels_)
