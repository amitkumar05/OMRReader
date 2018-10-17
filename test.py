import cv2
import sys
import math
from matplotlib import pyplot as plt
import numpy as np

def sobel(k):
    kernel1 = np.zeros((k,k),np.float32)/k**2
    for i in range(k):
        for j in range(k):
            if(i==0):
                kernel1[i][j]=-1
            if(i==k-1):
                kernel1[i][j]=1
    kernel2 = np.zeros((k,k),np.float32)/k**2
    for i in range(k):
        for j in range(k):
            if(j==0):
                kernel2[i][j]=-1
            if(j==k-1):
                kernel2[i][j]=1
    return kernel1,kernel2

khor,kvert=sobel(3)

img=cv2.imread('omr.jpg')
newimg1=cv2.filter2D(img,-1,khor)
cv2.imwrite("output.jpg",newimg1)
