# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:23:56 2023

@author: richie bao
迁移于：Region-Growing  https://github.com/Spinkoo/Region-Growing
"""
import cv2
import numpy as np
import random
import sys

class Stack():
    def __init__(self):
        self.item = []
        self.obj=[]
    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []

class Img_regionGrow():
  
    def __init__(self,th,im_path=None,im=None):
        if im is not None:
            self.im=im
        else:            
            self.readImage(im_path)
        self.h, self.w,_ =  self.im.shape
        self.passedBy = np.zeros((self.h,self.w), np.double)
        self.currentRegion = 0 
        self.iterations=0
        self.SEGS=np.zeros((self.h,self.w,3), dtype='uint8')
        self.stack = Stack()
        self.thresh=float(th)
    def readImage(self, img_path):
        self.im = cv2.imread(img_path,1)    

    def getNeighbour(self, x0, y0):
        neighbour = []
        for i in (-1,0,1):
            for j in (-1,0,1):
                if (i,j) == (0,0): 
                    continue
                x = x0+i
                y = y0+j
                if self.limit(x,y):
                    neighbour.append((x,y))
        return neighbour
    
    def ApplyRegionGrow(self):
        randomseeds=[[self.h/2,self.w/2],
            [self.h/3,self.w/3],[2*self.h/3,self.w/3],[self.h/3-10,self.w/3],
            [self.h/3,2*self.w/3],[2*self.h/3,2*self.w/3],[self.h/3-10,2*self.w/3],
            [self.h/3,self.w-10],[2*self.h/3,self.w-10],[self.h/3-10,self.w-10]
                    ]
        np.random.shuffle(randomseeds)
        for x0 in range (self.h):
            for y0 in range (self.w):
         
                if self.passedBy[x0,y0] == 0 and (int(self.im[x0,y0,0])*int(self.im[x0,y0,1])*int(self.im[x0,y0,2]) > 0) :  
                    self.currentRegion += 1
                    self.passedBy[x0,y0] = self.currentRegion
                    self.stack.push((x0,y0))
                    self.prev_region_count=0
                    while not self.stack.isEmpty():
                        x,y = self.stack.pop()
                        self.BFS(x,y)
                        self.iterations+=1
                    if(self.PassedAll()):
                        break
                    if(self.prev_region_count<8*8):     
                        self.passedBy[self.passedBy==self.currentRegion]=0
                        x0=random.randint(x0-4,x0+4)
                        y0=random.randint(y0-4,y0+4)
                        x0=max(0,x0)
                        y0=max(0,y0)
                        x0=min(x0,self.h-1)
                        y0=min(y0,self.w-1)
                        self.currentRegion-=1

        for i in range(0,self.h):
            for j in range (0,self.w):
                val = self.passedBy[i][j]
                if(val==0):
                    self.SEGS[i][j]=255,255,255
                else:
                    self.SEGS[i][j]=val*35,val*90,val*30
        if(self.iterations>200000):
            print("Max Iterations")
        print("Iterations : "+str(self.iterations))
        cv2.imshow("",self.SEGS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def BFS(self, x0,y0):
        regionNum = self.passedBy[x0,y0]
        elems=[]
        elems.append((int(self.im[x0,y0,0])+int(self.im[x0,y0,1])+int(self.im[x0,y0,2]))/3)
        var=self.thresh
        neighbours=self.getNeighbour(x0,y0)
        
        for x,y in neighbours:
            if self.passedBy[x,y] == 0 and self.distance(x,y,x0,y0)<var:
                if(self.PassedAll()):
                    break;
                self.passedBy[x,y] = regionNum
                self.stack.push((x,y))
                elems.append((int(self.im[x,y,0])+int(self.im[x,y,1])+int(self.im[x,y,2]))/3)
                var=np.var(elems)
                self.prev_region_count+=1
            var=max(var,self.thresh)
            
    def PassedAll(self):   
        return self.iterations>200000 or np.count_nonzero(self.passedBy > 0) == self.w*self.h

    def limit(self, x,y):
        return  0<=x<self.h and 0<=y<self.w
    
    def distance(self,x,y,x0,y0):
        return ((int(self.im[x,y,0])-int(self.im[x0,y0,0]))**2+(int(self.im[x,y,1])-int(self.im[x0,y0,1]))**2+(int(self.im[x,y,2])-int(self.im[x0,y0,2]))**2)**0.5

class Img_gray_regionGrow():
  
    def __init__(self,th,im_path=None,im=None,fill_val=0):
        if im is not None:
            self.im=im
        else:            
            self.readImage(im_path)
        self.fill_val=fill_val
        self.h, self.w=  self.im.shape
        self.passedBy = np.zeros((self.h,self.w), np.double)
        self.currentRegion = 0 
        self.iterations=0
        self.SEGS=np.zeros((self.h,self.w), dtype='uint8')
        self.stack = Stack()
        self.thresh=float(th)
    def readImage(self, img_path):
        self.im = cv2.imread(img_path,1)    

    def getNeighbour(self, x0, y0):
        neighbour = []
        for i in (-1,0,1):
            for j in (-1,0,1):
                if (i,j) == (0,0): 
                    continue
                x = x0+i
                y = y0+j
                if self.limit(x,y):
                    neighbour.append((x,y))
        return neighbour
    
    def ApplyRegionGrow(self):
        randomseeds=[[self.h/2,self.w/2],
            [self.h/3,self.w/3],[2*self.h/3,self.w/3],[self.h/3-10,self.w/3],
            [self.h/3,2*self.w/3],[2*self.h/3,2*self.w/3],[self.h/3-10,2*self.w/3],
            [self.h/3,self.w-10],[2*self.h/3,self.w-10],[self.h/3-10,self.w-10]
                    ]
        np.random.shuffle(randomseeds)
        for x0 in range (self.h):
            for y0 in range (self.w):
         
                if self.passedBy[x0,y0] == 0:  
                    self.currentRegion += 1
                    self.passedBy[x0,y0] = self.currentRegion
                    self.stack.push((x0,y0))
                    self.prev_region_count=0
                    while not self.stack.isEmpty():
                        x,y = self.stack.pop()
                        self.BFS(x,y)
                        self.iterations+=1
                    if(self.PassedAll()):
                        break
                    if(self.prev_region_count<8*8):     
                        self.passedBy[self.passedBy==self.currentRegion]=0
                        x0=random.randint(x0-4,x0+4)
                        y0=random.randint(y0-4,y0+4)
                        x0=max(0,x0)
                        y0=max(0,y0)
                        x0=min(x0,self.h-1)
                        y0=min(y0,self.w-1)
                        self.currentRegion-=1

        for i in range(0,self.h):
            for j in range (0,self.w):
                val = self.passedBy[i][j]
                if(val==0):
                    self.SEGS[i][j]=self.fill_val
                else:
                    self.SEGS[i][j]=val
        if(self.iterations>200000):
            print("Max Iterations")
        print("Iterations : "+str(self.iterations))
        #cv2.imshow("",self.SEGS)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
    def BFS(self, x0,y0):
        regionNum = self.passedBy[x0,y0]
        elems=[]
        elems.append(self.im[x0,y0])
        var=self.thresh
        neighbours=self.getNeighbour(x0,y0)
        
        for x,y in neighbours:
            if self.passedBy[x,y] == 0 and self.distance(x,y,x0,y0)<var:
                if(self.PassedAll()):
                    break;
                self.passedBy[x,y] = regionNum
                self.stack.push((x,y))
                elems.append(self.im[x,y])
                var=np.var(elems)
                self.prev_region_count+=1
            var=max(var,self.thresh)
            
    def PassedAll(self):   
        return self.iterations>200000 or np.count_nonzero(self.passedBy > 0) == self.w*self.h

    def limit(self, x,y):
        return  0<=x<self.h and 0<=y<self.w
    
    def distance(self,x,y,x0,y0):
        return ((int(self.im[x,y])-int(self.im[x0,y0]))**2)**0.5


if __name__=="__main__":  
    fn=r'C:\Users\richi\omen_richiebao\omen_code\Region-Growing-master\2.jpg'
    im=cv2.imread(fn,1)    
    # print(im)
    exemple =Img_gray_regionGrow(im=im[0],th=10)
    exemple.ApplyRegionGrow()
    print(exemple.SEGS.shape)
