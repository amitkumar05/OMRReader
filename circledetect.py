from __future__ import print_function
import cv2
import sys
import os
import math
#from matplotlib import pyplot as plt
import numpy as np
import copy
from scipy import arange
INF=1e18

def show(img,name="tmp.jpg"):
    cv2.imwrite(name,img)
    #if(name=="tmp.jpg"):
    #   os.system("open tmp.jpg")

def checkc(xc,yc,circles,dist):
    for (x,y,r) in circles:
        if( (x-xc)**2+(y-yc)**2<dist**2):
            return False
    return True


def hought(img,minref,minr,maxr,dist):
    thres,resthres=cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    edges = cv2.Canny(resthres,200,255)
    show(edges,"edges.jpg")
    radius=[minr,maxr,1]
    X,Y,c=img.shape
    data=np.array([[[0]*(radius[1]-radius[0]+1)]*Y]*X)


    # In[224]:

    for i in range(X):
        for j in range(Y):
            if(edges[i][j]!=0):
                for r in range(radius[0],radius[1],radius[2]):
                    for x in range(i-r,i+r):
                        if(x>=0 and x<X):
                            ysq=r**2-(x-i)**2
                            if(ysq>0):
                                y=ysq**0.5+j
                                y=int(y)
                                if(0<=y<Y):
                                    data[x][y][r-radius[0]]+=1

    maxv=-1e18
    ans=[0,0,0]
    dt=[]
    for i in range(X):
        for j in range(Y):
            for r in range(0,radius[1]-radius[0]+1,radius[2]):
                dt.append([data[i][j][r],i,j,r])

    dt=sorted(dt, key=lambda student: student[0],reverse=True)


    circles=[]


    for i in range(len(dt)):
        if(dt[i][0]<minref):
            break
        if(checkc(dt[i][2],dt[i][1],circles,dist)):
            circles.append([dt[i][2],dt[i][1],dt[i][3]+radius[0]])


    return circles




img=cv2.imread(sys.argv[1])
#def hought(img,minref,minr,maxr,dist):

circles=hought(img,50,96,103,50)
out=img.copy()
for (x, y, r) in circles:
    cv2.circle(out, (x, y), r, (0, 255, 0), 4)

show(out)
