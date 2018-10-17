from __future__ import print_function
import cv2
import sys
import os
import math
from matplotlib import pyplot as plt
import numpy as np
import copy
from scipy import arange
INF=1e18

def show(img):
    cv2.imwrite("tmp.jpg",img)
    os.system("xdg-open tmp.jpg")


#Circles
img1=cv2.imread("circle3.jpg")
thres,resthres=cv2.threshold(img1,150,255,cv2.THRESH_BINARY)
checkimg = cv2.cvtColor(resthres, cv2.COLOR_BGR2GRAY)
show(checkimg)
