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
            if i <= n and j<=m:
                window[i-x,j-y] = image[i,j]
            else:
                window[i-x,j-y] = 0 #padding out of bounds entry with 0s
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

train_filenames = os.listdir(path)

print(len(train_filenames))

im =  []

for j in range(len(train_filenames)):
    im = Image.open(path + train_filenames[j])
    im = ImageOps.grayscale(im)
    im = np.array(im)
    print(j)   
    print(im.shape)




pixel = 0,0  #x, y respectively or rows and column entry
  #height, width respectively


start = time.time()


#------------------------ Post Processing/ plotting -------------------------------

h = 230 #height    
w = 230 #width


[Dgm0,fraction_image] = fractional_lifespancurve(im,h,w,pixel)


end = time.time()

print("Time elapsed:", end-start)

print("This is the original image:",im)
print("This is the original size:", im.shape)

print("This is the window:",fraction_image)
print("This is the window size:", fraction_image.shape)


'''
plt.imshow(fraction_image)
plt.show()



#Uncomment to see the plot for the persistent diagram
plt.figure(1)
gudhi.plot_persistence_diagram(Dgm0)
plt.show()
'''





D0 = pc.Diagram(Dgm =Dgm0, globalmaxdeath = None, infinitedeath=None, inf_policy="keep")
dgm0 = D0.Betticurve(meshstart=0,meshstop=256,num_in_mesh=256)






#Uncomment to see the persistence curve for 0-dimension betti
#plt.plot(x, dgm0)
