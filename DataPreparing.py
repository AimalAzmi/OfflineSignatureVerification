# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 12:14:28 2018

@author: Manoochehr
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
#from SignatureCroping import getSignatureFromPage

#pathGen = "DatasetSig1/original/*/*.png"
#pathForg = "DatasetSig1/forgeries/*/*.png"

def DataPreparing(path):
        filenames = glob.glob(path)
        imagesX = [cv2.imread(img,cv2.IMREAD_GRAYSCALE) for img in filenames]
        y_all = []
        for file in filenames:
                _,imgY,_ = file.split("\\")
                y_all.append(imgY)
                #endfor                 
        X_all = []
        for img in imagesX:              
                img = 255 - img 
                img = cv2.resize(img,(220, 150))
                _,thresh1 = cv2.threshold(img,30,1,cv2.THRESH_BINARY) 
                img = img * thresh1                                 
                img = cv2.fastNlMeansDenoising(img,None,10,7,21)
                
                X_all.append(img)
                #endfor                
        y_all = np.asarray(y_all)
        X_all = np.asarray(X_all)        
        
        return X_all,y_all

#X_all,y_all = DataPreparing(pathGen)
#
#img2 = X_all[10,:,:]
#plt.imshow(img2)
#
##cv2.imshow('image',img2)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#
#
#from SignatureCroping import getSignatureFromPage
#img3 = getSignatureFromPage(img2)
#plt.imshow(img3)


#cv2.imshow('image',img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



