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

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

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

    return Dgm0,Dgm1, fraction_image

#--------------------Configuration---------------------- 

path = r"/home/nem2/Documents/HAM/HAM10000_images/"


'''
results_dir = os.path.dirname(__file__)

results_dir = os.path.join(results_dir, ID + 'Betticurves/')

if not os.path.isdir(results_dir):

    os.makedirs(results_dir)
#Creates a file directory depending on ID of image (useful for saving images)



im = []
Im = Image.open(path + ID +".jpg")
im = ImageOps.grayscale(Im)
im = np.array(im)


Im = Im.convert("RGBA")
colorimage= np.array(Im)
'''



HAM_filenames = os.listdir(path)

print(len(HAM_filenames))


x_1 = np.linspace(0,255,num=256)
#initializing plot 






'''
CL =np.load("clusteringlabels.npz")
CL = CL.f.arr_0  #turns dictionary to readable array
'''


c0= (0,0,0,255) #black 
c1= (0,0,255,100) #blue
c2= (0,255,0,100) #green
c3= (255, 0, 0, 100) #red
c4 = (186,85,211,100) #purple
c5=(255,255,255,255) #white
c6 = (153,76,0,100) #brown
c7 = (0,51,52,100) #mossgreen
c8 = (255,128,0,100) #orange
c9 = (0,51,102,100) #navyblue


Arrays = []

for k in range(100-4):

    plt.clf()

    im = []
    PC = []
    Pcurves = []

    x = 0
    y = 0
    #x, y respectively or rows and column entry
    #height, width respectively


    h = 20 #height    
    w = 20 #width
    s = 5 #stride


    
    ID = os.path.splitext(HAM_filenames[k+4])[0]

    results_dir = os.path.dirname(__file__)
    results_dir = os.path.join(results_dir, ID + "-pc" )
    if not os.path.isdir(results_dir):
       os.makedirs(results_dir)


    
    
    CL =np.load(r"/home/nem2/Documents/HAM/" + ID  + "-pc/clusteringlabels.npz")
    CL = CL.f.arr_0  #turns dictionary to readable array
    





    IM = Image.open(path + HAM_filenames[k+4])  #want to store IM for later clustering
    im = ImageOps.grayscale(IM)
    im = np.array(im)
    
    n,m = im.shape

    
    print("image shape:",im.shape)
    
    IM = IM.convert("RGBA")
    colorimage = np.array(IM)
    
    for p in range(n):
        for q in range(m):
            colorimage[p,q] = c0 #turns all images to black



    

    #------------------------ Post Processing/ plotting -------------------------------

    '''This will iterate the pixels
    to make our windows slide depending on increments
    '''

    i=0 #iteration

    Pcurves = []

    while y+w <= m:


        print("x=:",x)
        print("y=:",y)

        pixel = x,y

        [Dgm0,Dgm1,fraction_image]= fractional_lifespancurve(im, h, w , pixel)
        
        if Dgm1.size == 0:
            Dgm1 = np.zeros((2,2))


        #print("This is the window:",fraction_image)
        #print("This is the window size:", fraction_image.shape)

        ''' #uncomment this if you want to see the window images sliding
        plt.imshow(fraction_image)
        plt.show()
        '''
        print("Dgm1", Dgm1)
        D0 = pc.Diagram(Dgm =Dgm0, globalmaxdeath = None, infinitedeath=None, inf_policy="keep")
        D1 = pc.Diagram(Dgm = Dgm1, globalmaxdeath = None, infinitedeath= None, inf_policy="keep")

        B0 = D0.Betticurve(0,256,256)
        B1 = D1.Betticurve(0,256,256)

        G0 = D0.gaussian_Betti(meshstart=0,meshstop=256,num_in_mesh=256, spread=1)
        G1 = D1.gaussian_Betti(meshstart=0,meshstop=256,num_in_mesh=256, spread=1)
        
        
        L0 = D0.normalizedlifecurve(0,256,256)
        L1= D1.normalizedlifecurve(0,256,256)

        
        L0[np.any(np.isnan(L0)) == True] =0 
        L0[np.all(np.isfinite(L0)) == True] =0
        L1[np.any(np.isnan(L1)) == True] = 0
        L1[np.all(np.isfinite(L1)) == True] =0
        

        PC = np.hstack((B0,G0,L0, B1,G1,L1)) 
        Pcurves.append(PC)

        
     
        #uncomment this if you want to see the betticurves for windows
    

        #plt.savefig(results_dir + "ID:" + ID +" x_"+str(x) + "y_" + str(y)+".png")
        #print("betticurvearray:", dgm0)
    

        #np.save(results_dir + ID + "x_" +str(x) + "y_" + str(y) , dgm0)

        y = y+s

        if y+w > m :
            y=0
            x = x +s

        if x+h >= n:

            break


        #print("iteration count:", i)     
        i =i+1

           
                        
        if CL[i] == 0:

            colorimage[x,y] = c5
        
        elif CL[i] == 1:

            colorimage[x,y] = c0
        '''
        elif CL[j] == 2:
        
            colorimage[x,y] = c2
        
        elif CL[j] == 3:
            colorimage[x,y] = c3
        
        elif CL[j] == 4:
            colorimage[x,y] = c4
        
        elif CL[j] == 5:
            colorimage[x,y] = c6
        '''
    
    
    np.savez(results_dir + "/Pcurves", Pcurves)
    
    
    print("size of All PC  DATA:",Pcurves)


    clustering = KMeans(n_clusters=2,random_state=0).fit(Pcurves)
    


    #clustering = AgglomerativeClustering(n_clusters= None,distance_threshold = 1500).fit(BettiArrays)

    np.savez( results_dir  + "/clusteringlabels", clustering.labels_)
    

    fig = plt.figure()    
    plt.imshow(colorimage)
    #plt.show()
    fig.savefig(results_dir + "/colorimage.png")
    
    
    print(clustering.labels_)
    print(np.unique(clustering.labels_))
    print("count:", k)
