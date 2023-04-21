# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 07:20:41 2023

@author: richie bao
"""
from tkinter import (Tk, ttk, colorchooser, filedialog, messagebox, Label, 
                     LabelFrame,Button,LEFT,Scale,VERTICAL,Canvas,LEFT,
                     PhotoImage,NW,W,LEFT,StringVar)
import PIL.ImageGrab as ImageGrab
from PIL import Image,ImageTk,ImageDraw
import tempfile
import os
from pathlib import Path
import numpy as np
import datetime

if __package__:
    from .A2B import A2B_generator
else:
    from A2B import A2B_generator
    

from PIL import EpsImagePlugin
EpsImagePlugin.gs_windows_binary =r'C:\Program Files\gs\gs10.01.1\bin\gswin64c'

LC_color_dict={
    0: (0, 0, 0),
    1: (0, 197, 255),
    2: (0, 168, 132),
    3: (38, 115, 0),
    4: (76, 230, 0),
    5: (163, 255, 115),
    6: (255, 170, 0),
    7: (255, 0, 0,),
    8: (156, 156, 156),
    9: (0, 0, 0),
    10: (115, 115, 0),
    11: (230, 230, 0),
    12: (255, 255, 115),
    13: (197, 0, 255),
    }

LC_id2name={0:'None',
            1:'Water',
            2:'Emergent Wetlands', 
            3:'Tree Canopy',
            4:'Scrub/Shrub',
            5:'Low Vegetation',
            6:'Barren',
            7:'Impervious Structures',
            8:'Other Impervious',
            9:'Impervious Road',
            10:'Tree Canopy over Impervious Structure',
            11:'Tree Canopy over Other Impervious',
            12:'Tree Canopy over Impervious Roads',
            13:'Aberdeen Proving Ground'}

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

class Sketch_A2B(Tk):
    
    def __init__(self):
        super().__init__()
        self.initializeUI()
        self.width=512
        self.height=512        
        
    def initializeUI(self):
        self.title('Sketch A2B or B2A')
        self.minsize(500, 250)  # width, height
        self.geometry("1400x700+50+50")
        # self.config(background='white')
        self.setupWindow()
        
    def setupWindow(self):
        """ Set up the widgets."""
        title=Label(self, text="From LC Sketches to Images",font=('Helvetica',15),bd=10)
        title.grid(row=0,column=0,columnspan=2)
        
        self.pointer= "black"
        self.erase="white"
        
        self.pick_color=LabelFrame(self,text='Colors',font =('arial',15)) 
        self.pick_color.grid(row=1,rowspan=7*10,column=0) # column=0, rowspan=6,columnspan=2, padx=5, pady=5
        
        LC_color_hex_dict={k:rgb_to_hex(v) for k, v in LC_color_dict.items()}
        i=j=0
        for idx,color in LC_color_hex_dict.items():
            Button(self.pick_color,
                   bg=color,
                   width=15,
                   height=5,
                   text=LC_id2name[idx],
                   fg='white',
                   justify=LEFT,
                   wraplength=100,
                    # font='sans 10 bold',
                   command=lambda col=color:self.select_color(col)).grid(row=i,column=j)            
            i+=1
            if i==7:
                i=0
                j=1  
        # Erase Button and its properties          
        self.eraser_btn=Button(self,text="Eraser",command=self.eraser,width=9).grid(row=1,column=1)  
        # Reset Button to clear the entire screen
        self.clear_screen=Button(self,text="Clear Screen",command= lambda : self.canvas.delete('all'),width=9).grid(row=2,column=1) 
        # Save Button for saving the image in local computer
        self.save_btn=Button(self,text="ScreenShot",command=self.save_drawing,width=9).grid(row=3,column=1)          
        # Background Button for choosing color of the Canvas
        self.bg_btn=Button(self,text="Background",command=self.canvas_color,width=9).grid(row=4,column=1)  
        # pretrained model
        self.load_btn=Button(self,text="laod model",command=self.browseFiles,width=9).grid(row=5,column=1)     
        
        self.update_btn=Button(self,text="update G_Img",command=self.A2B,width=9,bg='red').grid(row=6,column=1) 
        
        #Creating a Scale for pointer and eraser size
        self.pointer_frame=LabelFrame(self,text='size',font=('arial',15,'bold')).grid(row=7,columns=1)    
        self.pointer_size=Scale(self.pointer_frame,orient=VERTICAL,from_ =48 , to =0, length=168,width=55)
        self.pointer_size.set(24)
        self.pointer_size.grid(row=7,column=1)            
        
        #Defining a background color for the Canvas 
        self.canvas=Canvas(self,bg='white',bd=5,height=512,width=512)     
        self.canvas.grid(row=1,column=2,rowspan=10)   
        #Bind the background Canvas with mouse click
        self.canvas.bind("<B1-Motion>",self.paint)   
        
        # self.container=self.canvas.create_rectangle(0, 0, 512, 512,) # outline='white'
        
        self.canvas4img=Canvas(self,bg='white',bd=5,height=512,width=512)  
        self.canvas4img.grid(row=1,column=3,rowspan=10)
        temp_fn=r'C:\Users\richi\omen_richiebao\omen_temp\22.jpg'
        self.im=im = ImageTk.PhotoImage(Image.open(temp_fn))
        self.img_container=self.canvas4img.create_image((0,0),anchor=NW, image=im)
        
        # self.var_time = StringVar()
        self.time_label=Label(self, text="placeholder",width=20,anchor=W,justify=LEFT) # textvariable=self.var_time,
        self.time_label.grid(row=12,column=2,sticky = W)
        self.update_time()
        
        self.A2B()
        
        
    # Function for defining the eraser
    def eraser(self):
        self.pointer= self.erase   
        
    # Function for saving the image file in Local Computer
    def save_drawing(self):
        self.canvas.update()   
  
        with tempfile.TemporaryFile(mode='w+b',delete=False) as fp:  
            self.canvas.postscript(file=fp.name, colormode='color')
            im=Image.open(fp.name)               
            im=im.resize((512,512))
            file_ss=filedialog.asksaveasfilename(defaultextension='.png')            
            im.save(file_ss,'png')   
            
    # Function for choosing the background color of the Canvas    
    def canvas_color(self):
        color=colorchooser.askcolor()
        self.canvas.configure(background=color[1])
        self.erase= color[1]       
        
    # Paint Function for Drawing the lines on Canvas
    def paint(self,event):       
        x1,y1 = (event.x-2), (event.y-2)  
        x2,y2 = (event.x+2), (event.y+2)  

        self.canvas.create_oval(x1,y1,x2,y2,fill=self.pointer,outline=self.pointer,width=self.pointer_size.get())  
        
    # Function for choosing the color of pointer  
    def select_color(self,col):
        self.pointer = col           
        
    def browseFiles(self):
        self.model_fn = filedialog.askopenfilename(initialdir = "/",
                                              title = "Select a File",
                                              filetypes = (("pth","*.pth*"),
                                                           ("ckp", "*.ckp*"),                                                           
                                                           ("all files","*.*")))        
    
    def update_time(self):
        currentTime = datetime.datetime.now()
        # print('---',self.time_label) 
        self.time_label['text']=currentTime
        # self.var_time=currentTime
        self.after(60*10,self.update_time)
        
    def A2B(self):
        try:
            self.canvas.update()     
            with tempfile.TemporaryFile(mode='w+b',delete=False) as fp:  
                self.canvas.postscript(file=fp.name, colormode='color')
                im=Image.open(fp.name)               
                im=im.resize((512,512))
                
            with tempfile.TemporaryFile(mode='w+b',delete=False) as fp: 
                im.save(fp.name+'.png','png')
                
                AorB,BorA=A2B_generator(self.model_fn,fp.name+'.png')    
                
                self.g_img=g_img = ImageTk.PhotoImage(BorA)
                # print(self.g_img)                
                self.canvas4img.itemconfig(self.img_container,image=self.g_img)
                # self.canvas4img.create_image((0,0),anchor=NW, image=g_img)               

        except:
            pass
        
        # self.after(60*10,self.A2B)
        
        
if  __name__  ==  "__main__":
    app=Sketch_A2B()
    app.mainloop()