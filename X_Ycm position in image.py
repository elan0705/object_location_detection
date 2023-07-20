#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import time


# In[ ]:
#to find the locate the object in an image and get its location in cm by subtracting the foreground and background

cap = cv2.VideoCapture(0)

#field of view
cm_pxX = 11.3/640.0
cm_pxY = 11.3/640.0


#allihning camera base frame axis and image origin axis on top-left corner
R180_X = [ [1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)] ] 
Rad = (-94.0/180)*np.pi
RZ= [ [np.cos(Rad), -np.sin(Rad),0], [np.sin(Rad), np.cos(Rad), 0], [0, 0, 1] ]
R0_C =np.dot(R180_X, RZ)

d0_c = [[-1.8],[-0.3],[0]]

H0_c = np.concatenate((R0_C, d0_c),1)
H0_c = np.concatenate((H0_c,[[0,0,0,1]]),0)
while(1):
    _,frame = cap.read()
    
    gr1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
    cv2.imshow('background',gr1)
   
    k = cv2.waitKey(5)
    if k == 27:
        break

while(1):
    _,frame = cap.read()
    
    gr2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
    cv2.imshow('foreground',gr2)
    
    diff = np.absolute(np.matrix(np.int16(gr1)) - np.matrix(np.int16(gr2)))
    diff[diff>255]=255
    diff=np.uint8(diff)
    cv2.imshow('diff', diff)
    
    BW = diff
    BW[BW<=100]=0
    BW[BW>100]=1
    
    
    col_sum = np.matrix(np.sum(BW,0))
    col_no = np.matrix(np.arange(640))
    col_mul = np.multiply(col_sum,col_no)
    tot = np.sum(col_mul)
    ToT = np.sum(np.sum(BW))
    col_loc = tot/ToT
    X = col_loc*cm_pxX
   
    row_sum = np.matrix(np.sum(BW,1))
    row_sum = row_sum.transpose()
    row_no = np.matrix(np.arange(480))
    row_mul = np.multiply(row_sum,row_no)
    tot = np.sum(row_mul)
    ToT = np.sum(np.sum(BW))
    row_loc = tot/ToT
    Y = row_loc*cm_pxY
   
    pnt = [[X],[Y],[0],[1]]
    P0 = np.dot(H0_c,pnt)
    X0=P0[0]
    Y0 =P0[1]
    
    print('X'+'='+ str(X0))
    
    print('Y'+'='+ str(Y0))
    
    k = cv2.waitKey(5)
    if k == 27:
        break



while(1):
    _,frame = cap.read()
    
    
    k = cv2.waitKey(5)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




