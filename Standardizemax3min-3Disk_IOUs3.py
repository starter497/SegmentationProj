import os

#import pandas as pd

import numpy as np

import matplotlib

import shutil

from PIL import Image, ImageOps

from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings("ignore")

usetex = matplotlib.checkdep_usetex(True) #I dont have latex)





path1 =  r"/home/nem2/Documents/HAM/HAM10000_segmentations_lesion_tschandl/" 

path2 = r"/home/nem2/Documents/HAM/Standardizemax3min-3DiskIDs3-pc/"

path3 = r"/home/nem2/Documents/HAM/HAM10000_images/"

path4 = r"/home/nem2/Documents/HAM/"

Mymaskfiles = os.listdir(path2) # This is the mask that we generated and want to consider
humanmaskfiles = os.listdir(path1) #there are 1000files.  We need to select the correct 100 ID files.
imagefiles = os.listdir(path3) #images

IOU_total=[]
IOU_score_pass = 0
IOU_A= 0
IOU_totalM0= []
IOU_totalM1 = []
IOU_totalM2 = []



#--------------- Centrality function ----------------



# h,k are the center of the image
def centrality(image):
    n,m = image.shape
    h = n/2
    k = m/2
    centrality_score = []
    for x in range(n):
        for y in range(m):
            center_score= np.square(x-h) + np.square(y-k) 
            centrality_score.append(center_score) 
    return np.sum(centrality_score)



#----------------------------------------------------



#---------------------- Least intersection with the edge function --------------

def count_edge_pixels(image):
    
    n,m = image.shape
    edge = 21
    top = np.sum(image[edge, edge:m-edge])

    bottom = np.sum(image[n-edge, edge:m-edge])

    left = np.sum(image[edge:n-edge, edge])

    right = np.sum(image[edge:n-edge, m-edge])
    
    return top + bottom + left + right


#---------------------------------------------



for j in range(len(humanmaskfiles)): 

    for i in range(len(Mymaskfiles)):
        #we are scanning the 1000masks for the correct 100 masks
        
        #shutil.copy(path2+Mymaskfiles[i]+"/s3diskmask0.png", path4 + "MASK0/"+Mymaskfiles[i][0:12]+".png")
        #shutil.copy(path2+Mymaskfiles[i]+"/s3diskmask1.png", path4 + "MASK1/"+Mymaskfiles[i][0:12]+".png")
        #shutil.copy(path2+Mymaskfiles[i]+"/newmask2.png",path4 + "MASK2/"+Mymaskfiles[i][0:12]+".png")
        



        #if Mymaskfiles[i][0:12] == imagefiles[j][0:12]:
            
            #shutil.copy(path3 + imagefiles[j],path4+"image_ID/")
            
            
        if Mymaskfiles[i][0:12] == humanmaskfiles[j][0:12]: #Retrieves ID string
            
            #shutil.copy(path2+Mymaskfiles[i]+"/"+Mymaskfiles[i][0:12]+"_segmentation.png", path4 + "HAMMASK/")

            #shutil.copy(path1+humanmaskfiles[j],path2+ Mymaskfiles[i])
            

            HM = Image.open(path1 + humanmaskfiles[j])
            HM = ImageOps.grayscale(HM)
            HM = np.array(HM)   
            
            #print("human array:",HM)
            

           # MyMask =np.load(path2 + Mymaskfiles[i]+ "/PixelArray.npz")
           # MM = MyMask.f.arr_0  #turns dictionary to readable array
            
            
            arr0 = np.load(path2 + Mymaskfiles[i] + "/s3diskArr0.npz")
            arr0 = arr0.f.arr_0
            
            arr1 = np.load(path2 + Mymaskfiles[i] + "/s3diskArr1.npz")
            arr1 = arr1.f.arr_0
            


            '''

            #centrality score method
            if centrality(arr0) < centrality(arr1):
                arr2 = arr0
                
            else:
                arr2 = arr1
            '''


            # ------------------  Edge intersection method -------------------
            


            if count_edge_pixels(arr0) < count_edge_pixels(arr1):
                arr2 = arr0
            else:
                arr2 = arr1

            fig = plt.figure()    
            plt.imshow(arr2,cmap= 'gray')
            #plt.show()
            fig.savefig(path2 + Mymaskfiles[i] + "/s3diskmask2.png")
            


            # ----------------------------------------------------------------







            #Mask2 = np.load(path2 + Mymaskfiles[i]+ "/diskArr2.npz")
            #Mask2 = Mask2.f.arr_0
            
            #Inter_value = MM[(MM == 255) & ( HM == 255)]

            Inter_valueM0 = arr0[ (arr0 == 255) & ( HM == 255)]
            Inter_valueM1 = arr1[ (arr1 == 255) & (HM == 255)]
            
            Inter_valueM2 = arr2[(arr2 == 255) & (HM == 255)]
            
                #retreive a list for the  components of the array that satisfies
                #both conditions (our pixel is white and their pixel is white)


            

            #Intersection = len(Inter_value)         
            Intersection_M0 = len(Inter_valueM0)
            Intersection_M1 = len(Inter_valueM1)
        
            Intersection_M2 = len(Inter_valueM2)
            
                #retreives the length of the list.
                #Also the number of intersection values.



            #union_value = MM[ (MM == 255) | (HM == 255)]
            union_valueM0 = arr0[(arr0 == 255) | (HM ==255)]
            union_valueM1 = arr1[(arr1 == 255) | (HM == 255)]
            
            union_valueM2 = arr2[(arr2 == 255) | (HM == 255)]
                #retreives a list of components in our array that has
                #either a white pixel in our mask or 
                #a white pixel in the dataset given mask.
        
            
            #union = len(union_value)
            unionM0 = len(union_valueM0)
            unionM1 = len(union_valueM1)
        
            unionM2 = len(union_valueM2)
            
                #retrieves the length of the list.
                #Also the number of union values.


            #IOU_score = Intersection / union 
            IOU_score_M0 = Intersection_M0/ unionM0
            IOU_score_M1 = Intersection_M1/ unionM1
            
            IOU_score_M2 = Intersection_M2/unionM2
                #different scores for different masks
            
            #print("%s & %s\\\\" %(Mymaskfiles[i][5:12],IOU_score_M2))
        
            #print("\\hline")
            
            #print("IOU score M0", IOU_score_M0)

            #print("IOU score M1", IOU_score_M1)


            #if IOU_score_M0 < IOU_score_M1:
                #IOU_score_M0 = IOU_score_M1

            #print("IOU score M0", IOU_score_M0)
            

            print("IOU SCORE M2", IOU_score_M2)
            #print("ID:",Mymaskfiles[i][0:12])
            
            #print("IOU_score:" ,IOU_score)
            #print("IOU_score_OM:", IOU_score_OM)
                
            #IOU_total.append(IOU_score)
            IOU_totalM0.append(IOU_score_M0)
            IOU_totalM1.append(IOU_score_M1)
            
            IOU_totalM2.append(IOU_score_M2)
            
            if IOU_score_M2 > 0.7 :
                IOU_score_pass = IOU_score_pass + 1
                
            if IOU_score_M2 > 0.9 : 
                IOU_A = IOU_A +1
            
            #np.savetxt(r"/home/nem2/Documents/HAM/" + Mymaskfiles[i] + "IOU_score", np.array(IOU_score))
            
print("IOU pass scores:",IOU_score_pass)

print("IOU A scores:", IOU_A)

#IOU_AVG = sum(IOU_total)/len(IOU_total)

IOU_AVGM2 = sum(IOU_totalM2)/len(IOU_totalM2)

#IOU_AVGM0 = sum(IOU_totalM0)/len(IOU_totalM0)

#print("IOU AVERAGE:", IOU_AVG)
print("IOU_M2 AVERAGE:", IOU_AVGM2)

#print("IOU_M0 AVERAGE:", IOU_AVGM0)
#np.savetxt(r"/home/nem2/Documents/HAM/IOU_AVG", np.array(IOU_AVG))

