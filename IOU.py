import os

#import pandas as pd

import numpy as np

import matplotlib

from PIL import Image, ImageOps

from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings("ignore")

usetex = matplotlib.checkdep_usetex(True) #I dont have latex)





path1 =  r"/home/nem2/Documents/HAM/HAM10000_segmentations_lesion_tschandl/" 

path2 = r"/home/nem2/Documents/HAM/ID-pc/"

Mymaskfiles = os.listdir(path2) # This is the mask that we generated and want to consider
humanmaskfiles = os.listdir(path1) #there are 1000files.  We need to select the correct 100 ID files.

IOU_total=[]
IOU_score_pass = 0
IOU_A= 0
IOU_totalM0= []
IOU_totalM1 = []

for j in range(len(humanmaskfiles)): 

    for i in range(len(Mymaskfiles)):
        #we are scanning the 1000masks for the correct 100 masks
        

        if Mymaskfiles[i][0:12] == humanmaskfiles[j][0:12]: #Retrieves ID string

            HM = Image.open(path1 + humanmaskfiles[j])
            HM = ImageOps.grayscale(HM)
            HM = np.array(HM)   
            
            #print("human array:",HM)
            

            MyMask =np.load(path2 + Mymaskfiles[i]+ "/ourPixelArray.npz")
            MM = MyMask.f.arr_0  #turns dictionary to readable array
            
            Othermask= np.load(path2 + Mymaskfiles[i] + "/PixelArray.npz")
            OM = Othermask.f.arr_0

            Mask0 = np.load(path2 + Mymaskfiles[i] + "/Array0.npz")
            Mask0 = Mask0.f.arr_0
            
            Mask1 = np.load(path2 + Mymaskfiles[i] + "/Array1.npz")
            Mask1 = Mask1.f.arr_0


            Inter_value = MM[(MM == 255) & ( HM == 255)]
            Inter_value_OM = OM[(OM == 255) & (HM == 255 )]
            Inter_valueM0 = Mask0[ (Mask0 == 255) & ( HM == 255)]
            Inter_valueM1 = Mask1 [ (Mask1 == 255) & (HM == 255)]

                #retreive a list for the  components of the array that satisfies
                #both conditions (our pixel is white and their pixel is white)


            

            Intersection = len(Inter_value)            
            Intersection_OM = len(Inter_value_OM)
            Intersection_M0 = len(Inter_valueM0)
            Intersection_M1 = len(Inter_valueM1)

                #retreives the length of the list.
                #Also the number of intersection values.



            union_value_OM = OM[ (OM == 255) | (HM == 255)]
            union_value = MM[ (MM == 255) | (HM == 255)]
            union_valueM0 = Mask0[(Mask0 == 255) | (HM ==255)]
            union_valueM1 = Mask1[(Mask1 == 255) | (HM == 255)]
                
                #retreives a list of components in our array that has
                #either a white pixel in our mask or 
                #a white pixel in the dataset given mask.

            union_OM = len(union_value_OM)
            union = len(union_value)
            unionM0 = len(union_valueM0)
            unionM1 = len(union_valueM1)

                #retrieves the length of the list.
                #Also the number of union values.


            IOU_score = Intersection / union
            IOU_score_OM = Intersection_OM / union_OM
            IOU_score_M0 = Intersection_M0/ unionM0
            IOU_score_M1 = Intersection_M1/ unionM1
                
                #different scores for different masks

            print("ID:", Mymaskfiles[i][0:12])
            print("IOU score M0", IOU_score_M0)

            print("IOU score M1", IOU_score_M1)


            if IOU_score_M0 < IOU_score_M1:
                IOU_score_M0 = IOU_score_M1

            #print("ID:",Mymaskfiles[i][0:12])
            
            #print("IOU_score:" ,IOU_score)
            #print("IOU_score_OM:", IOU_score_OM)
                
            IOU_total.append(IOU_score)
            IOU_totalM0.append(IOU_score_M0)
            IOU_totalM1.append(IOU_score_M1)


            if IOU_score > 0.5 :
                IOU_score_pass = IOU_score_pass + 1
                
            if IOU_score > 0.9 : 
                IOU_A = IOU_A +1

            #np.savetxt(r"/home/nem2/Documents/HAM/" + Mymaskfiles[i] + "IOU_score", np.array(IOU_score))

#print("IOU pass scores:",IOU_score_pass)

#print("IOU A scores:", IOU_A)

IOU_AVG = sum(IOU_total)/len(IOU_total)
IOU_AVGM0 = sum(IOU_totalM0)/len(IOU_totalM0)
#IOU_AVGM1 = sum(IOU_totalM1)/len(IOU_totalM1)

#print("IOU AVERAGE:", IOU_AVG)
print("IOU_M0 AVERAGE:", IOU_AVGM0)
#np.savetxt(r"/home/nem2/Documents/HAM/IOU_AVG", np.array(IOU_AVG))

