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

# -------------------- README ----------------------
'''

Hi!

if you are soon to run this file there are a couple of 
things to be aware of.  You will need to run this code at minimum 2 times.

The first run will run the while loop of the stride of windows whilst generating
the persistence curves. 

The main things you will need for the first run is to comment out any
"CL" that appears.  This is a array that is 
can only be called on after the first run 
has been made. This should already be done for you. Double check to be sure.

After the first run, you may comment a portion of the "STRIDE Process".
Detailed instruction will be given in the code. This will save a lot of time.

You now have a saved file labeled as clusterlabels.npz. Suppose for
the stride process you have generated n windows for an image.
The clusterlabels.npz is a vector with n dimensions. 
Each entry in that vector denotes the
Cluster class/label.

We will still iterate the while loop for the pixels
but each window generated will correspond to the clusterlabelz.npz component.
Hence, depending on the cluster label, we can fill in our image with white
pixels for the windows respecively during the while loop process.


Be sure to uncomment any CL notation in the code. You
should also uncomment the plot figures given at the very end of the code.
You may any saving procedure.

'''








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

                #we discuss this earlier but afterwards
                #I realize the "and" statement is justified 
                #since we start with an array of 0 and later 
                #add values.  Hence the "or" case would be zeroed out 
                #regardless.

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

#This was the procedure used for a single image.

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


CL =np.load("clusteringlabels.npz")
CL = CL.f.arr_0  #turns dictionary to readable array
'''



HAM_filenames = os.listdir(path)

print(len(HAM_filenamenames))

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

for k in range(100):

    #plt.clf()

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


    
    ID = os.path.splitext(HAM_filenames[k])[0]

    results_dir = os.path.dirname(__file__)
    results_dir = os.path.join(results_dir, ID + "-pc" )
    if not os.path.isdir(results_dir):
       os.makedirs(results_dir)

        #creates the directory for each image with named ID-pc 
    
    
    CL =np.load(r"/home/nem2/Documents/HAM/" + ID  + "-pc/clusteringlabels.npz")
    CL = CL.f.arr_0  #turns dictionary to readable array
        
        #loads our array in npz format to a readable array. 
        #Must be commented out initially until labels from clustering
        #have been saved
    



    IM = Image.open(path + HAM_filenames[k])  #want to store IM for later clustering
    im = ImageOps.grayscale(IM)
    im = np.array(im)
    
    n,m = im.shape

    
    #print("image shape:",im.shape)
    
    IM = ImageOps.grayscale(IM)
    IM = np.array(IM)

    #IM = IM.convert("RGBA")
    #colorimage = np.array(IM)
    
    for p in range(n):
        for q in range(m):
            IM[p,q] = 0 #turns all images to a black canvas



    

    #------------------------ STRIDE process -------------------------------

    '''
    This will iterate the pixels
    to make our windows slide depending on increments
    '''


    i=0 #iteration

    Pcurves = []

    while y+w <= m: 
        
        #while our window is not comletely to the right of the image


        #print("x=:",x)
        #print("y=:",y)

        pixel = x,y


        #---------------------------------------------------------------------
       
        #This procedure only needs to be done once.
        #Afterwards feel free to comment it out
    

        [Dgm0,Dgm1,fraction_image]= fractional_lifespancurve(im, h, w , pixel)
        
        if Dgm1.size == 0:
            Dgm1 = np.zeros((2,2))

                #pads any empty array given by the Dgm1
                #because lack of 1-dim features

        #print("This is the window:",fraction_image)
        #print("This is the window size:", fraction_image.shape)
        
        


        #---------------------------------------------------------------
        #uncomment this if you want to see the window images sliding to visually
        #see the algorithm.  

        
        #plt.imshow(fraction_image)
        #plt.show()
        #----------------------------------------------------------------

        

        
        D0 = pc.Diagram(Dgm =Dgm0, globalmaxdeath = None, infinitedeath=None, inf_policy="keep")
        D1 = pc.Diagram(Dgm = Dgm1, globalmaxdeath = None, infinitedeath= None, inf_policy="keep")
            #Using the persistent diagram information for curve summaries



        G0 = D0.gaussian_Betti(meshstart=0,meshstop=256,num_in_mesh=256, spread=1)
        G1 = D1.gaussian_Betti(meshstart=0,meshstop=256,num_in_mesh=256, spread=1)
        
        L0 = D0.normalizedlifecurve(0,256,256)
        L1= D1.normalizedlifecurve(0,256,256)

        
        L0[np.any(np.isnan(L0)) == True] =0 
        L0[np.all(np.isfinite(L0)) == True] =0
        L1[np.any(np.isnan(L1)) == True] = 0
        L1[np.all(np.isfinite(L1)) == True] =0
        
        
        PC = np.hstack((G0,L0,G1,L1))   #Each row is a sample curve
        Pcurves.append(PC)  #Turns for loop into a long vector
        
        
     
        #uncomment this if you want to save the betticurves for windows
    
        #plt.savefig(results_dir + "ID:" + ID +" x_"+str(x) + "y_" + str(y)+".png")
        #print("betticurvearray:", dgm0)
        #np.save(results_dir + ID + "x_" +str(x) + "y_" + str(y) , dgm0)







         # Remember the above script only needs to be done on the first run 
        #--------------------------------------------------------------------







        y = y+s  #as long as our pixel does not pass the edge, we will continue
        #the stride 

        if y+w > m :
            y=0
            x = x +s

                # Once our pixel is passed the edge, we will reset 
                our horizontal pixel but add a stride down

        if x+h >= n:


            break
                # we continue this until we cannot stride down anymore



            
    

        #print("iteration count:", i)     
        i =i+1

                   
       '''
        #This must be uncommented AFTER running once to get data saved

        #if CL[i] == 0 :
            
            #IM[x,y] = 0
                #comment out the either cluster 0 or 1 depending on our mask

        if CL[i] == 0:

            IM[x:x+h,y:y+h] = 255  #all values in the window will be white


        '''

        '''

        #This is was initially for agglomerative method
        #consisting of more than 2 clusters


        elif CL[j] == 2:
        
            colorimage[x,y] = c2
        
        elif CL[j] == 3:
            colorimage[x,y] = c3
        
        elif CL[j] == 4:
            colorimage[x,y] = c4
        
        elif CL[j] == 5:
            colorimage[x,y] = c6
        '''
    



    #---------------  Can comment after first run -------------


    # Only need to uncomment on the first run of the code.  Afterwards 
    #comment out later.  Choose which clustering method 
    #is preferred for the first run



    np.savez(results_dir + "/Pcurves", Pcurves)
    
    
    #print("size of All PC  DATA:",Pcurves)


    clustering = KMeans(n_clusters=2,random_state=0).fit(Pcurves)
    


    #clustering = AgglomerativeClustering(n_clusters= None,distance_threshold = 1500).fit(BettiArrays)

    np.savez( results_dir  + "/clusteringlabels", clustering.labels_)
    
    


    # Be sure to run this above for the first run at least

    # ------------------------------------------------------------

    '''
    #This is for the mask and its pixel array
    #Be sure to run this AFTER the first run


    fig = plt.figure()    
    plt.imshow(IM,cmap= 'gray')
    #plt.show()
    fig.savefig(results_dir + "/newmask1.png")

    np.savez(results_dir +"/Array1", IM)

    '''
    
    print(clustering.labels_)
    print(np.unique(clustering.labels_)) #number of distinctive clusters
    print("count:", k)
