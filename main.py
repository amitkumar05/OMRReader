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

def show(img,name):
    cv2.imwrite("out/"+name,img)
    #os.system("xdg-open tmp.jpg")

def checkc(xc,yc,circles,dist):
    for (x,y,r) in circles:
        if( (x-xc)**2+(y-yc)**2<dist**2):
            return False
    return True


def hought(img,minref,minr,maxr,dist):
    thres,resthres=cv2.threshold(img,130,255,cv2.THRESH_BINARY)
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





#Circles
img1=cv2.imread(sys.argv[1])
thres,resthres=cv2.threshold(img1,150,255,cv2.THRESH_BINARY)
checkimg = cv2.cvtColor(resthres, cv2.COLOR_BGR2GRAY)
show(checkimg,"binarythresholded.jpg")
#img1=img
gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


circles = cv2.HoughCircles(gray_image, cv2.cv.CV_HOUGH_GRADIENT, 11,25,minRadius=1,maxRadius=20)
output = img1.copy()
if circles is not None:
	circles = np.round(circles[0, :]).astype("int")
	for (x, y, r) in circles:
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
show(output,"circles.jpg")


#def hought(img,minref,minr,maxr,dist):
#circles=hought(img1,7,7,12,35)
# output = img1.copy()
# for (x, y, r) in circles:
# 	cv2.circle(output, (x, y), r, (0, 255, 0), 4)
# 	cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
# show(output,"circles.jpg")


# In[193]:

#interpolate
def fn(point,m,c,cdiff):
    ans=1
    for i in range(4):
        ans*=(m*point[1]-point[0]-c-cdiff*i)
    return abs(ans)

def lineinter(data):
    #c -60 -120 cdiff -40 -60
    m,c,cdiff=(-10,10,1),(-120,-60,1),(-60,-40,1)
    mins=INF
    ans=[0,0,0]
    for mv in arange(m[0],m[1],m[2]):
        for cv in arange(c[0],c[1],c[2]):
            for cdiffv in arange(cdiff[0],cdiff[1],cdiff[2]):
                score=0
                mvv=math.tan((mv*math.pi)/180)
                for point in data:
                    score+=fn(point,mvv,cv,cdiffv)
                if(mins>score):
                    mins=score
                    ans[0]=mvv
                    ans[1]=cv
                    ans[2]=cdiffv
    return ans



data=copy.copy(circles)
ans=lineinter(data)

def visual(img,ans):
    outp=img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p=[j,i]
            if(fn(p,ans[0],ans[1],ans[2])<400000):
                outp[i][j][0]=outp[i][j][1]=outp[i][j][2]=0
    return outp

outp=visual(img1,ans)
show(outp,"lines.jpg")

outp2=visual(output,ans)
show(outp2,"linescircles.jpg")


r=30

def isunique(points,tmp):
    for i in range(len(points)):
        if((points[i][0]-tmp[0])**2 + (points[i][1]-tmp[1])**2 < r*r):
            return 0
    return 1

def find_points(circles,m,c,cdiff):
    points = []
    points1 = []
    points2 = []
    points3 = []
    points4 = []
    print(m,c,cdiff)
    for i in range(len(circles)):
         if(isunique(points,circles[i])):
#          if(1):
            a = -1
            b=m
            p=circles[i][0]
            q=circles[i][1]
            for l in xrange(0,4):

                cc=c+l*cdiff

                c_new=q+m*p
                x=((-cc+c_new*m)/(m*m+1))
                y=c_new-m*x
                if(isunique(points,[x,y])):
                    points.append([x,y,0])
                    if(l==0):
                        points1.append([x,y,0])
                    elif(l==1):
                        points2.append([x,y,0])
                    elif(l==2):
                        points3.append([x,y,0])
                    elif(l==3):
                        points4.append([x,y,0])
                #print([x,y])
    return points,points1,points2,points3,points4


# In[229]:

points,points1,points2,points3,points4=find_points(circles,ans[0],ans[1],ans[2])



final=img1.copy()
points=sorted(points, key=lambda student: student[1])   # sort by age

for (x, y,r) in points:
    cv2.circle(final, (int(x),int(y)), 9, (0, 255, 0), 4)

show(final,"circlesestimated.jpg")


# In[234]:

points1=sorted(points1, key=lambda student: student[1])
points2=sorted(points2, key=lambda student: student[1])
points3=sorted(points3, key=lambda student: student[1])
points4=sorted(points4, key=lambda student: student[1])
pointsarr=[]
for i in range(len(points1)):
    pointsarr.append([points1[i],points2[i],points3[i],points4[i]])


# In[269]:

a=10
th=4*a*a*200
for i in range(len(pointsarr)):
    for j in range(len(pointsarr[i])):
        sum=0
        for l in xrange(int(pointsarr[i][j][0]-a),int(pointsarr[i][j][0]+a)):
            for k in xrange(int(pointsarr[i][j][1]-a),int(pointsarr[i][j][1]+a)):
                sum=sum+checkimg[k][l]
        if(sum < th):
            pointsarr[i][j][2]=1
        else:
            pointsarr[i][j][2]=0




# In[270]:

final2=img1.copy()

for i in range(len(pointsarr)):
    for j in range(len(pointsarr[i])):
        if(pointsarr[i][j][2]==1):
            cv2.circle(final2, (int(pointsarr[i][j][0]),int(pointsarr[i][j][1])), 9, (0, 255, 0), 4)
            print(i+1,j+1)
show(final2,"black_detector.jpg")
