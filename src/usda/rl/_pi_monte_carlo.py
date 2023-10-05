# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:28:55 2023

@author: richie bao

transfered and updated: Value of Pi using Monte Carlo – PYTHON PROGRAM，https://www.bragitoff.com/2021/05/value-of-pi-using-monte-carlo-python-program/
"""
# Author: Manas Sharma
# Website: www.bragitoff.com
# Email: manassharma07@live.com
# License: MIT
# Value of Pi using Monte carlo
 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
 
def pi_using_MonteCarlo(nTrials=int(10E4)): 
    # Input parameters
    radius = 1
    #-------------
    # Counter for the points inside the circle
    nInside = 0
     
    # Generate points in a square of side 2 units, from -1 to 1.
    XrandCoords = np.random.default_rng().uniform(-1, 1, (nTrials,))
    YrandCoords = np.random.default_rng().uniform(-1, 1, (nTrials,))
     
    for i in range(nTrials):
        x = XrandCoords[i]
        y = YrandCoords[i]
        # Check if the points are inside the circle or not
        if x**2+y**2<=radius**2:
            nInside = nInside + 1
    
    area = 4*nInside/nTrials
    print("Value of Pi: ",area)
    
def pi_using_MonteCarlo_anim(nTrials=int(10E4),figsize=(16, 8),step=100,interval=100):
    # Input parameters
    radius = 1
    #-------------
    # Counter for the pins inside the circle
    nInside = 0
    # Counter for the pins dropped
    nDrops = 0
     
    # Generate points in a square of side 2 units, from -1 to 1.
    XrandCoords = np.random.default_rng().uniform(-1, 1, (nTrials,))
    YrandCoords = np.random.default_rng().uniform(-1, 1, (nTrials,))    
    
    fig,(ax_lines,ax_scatter)=plt.subplots(1,2,figsize=figsize, dpi=120)
    
    # Some checks so the legend labels are only drawn once
    isFirst1 = True
    isFirst2 = True
     
    # Some arrays to store the pi value vs the number of pins dropped
    piValueI = []
    nDrops_arr = []
     
    # Some arrays to plot the points
    insideX = []
    outsideX = []
    insideY = []
    outsideY = []    

    line,=ax_lines.plot([],[],lw=2)
    # scatter=ax_scatter.scatter([],[],c='pink',s=50,label='Drop inside')
    scatter=ax_scatter.scatter([],[],c='pink',s=10,label='Drop inside')
    line_scatter=[line,scatter]
    text_scatter=ax_scatter.text(-0.9,1.1,'',fontsize=15)

    ax_lines.set_xlim(0,nTrials)
    ax_lines.set_ylim(0,4)
    ax_scatter.set_xlim(-1,1)
    ax_scatter.set_ylim(-1,1)

    def pi_cal(j):   
        nonlocal nDrops
        nonlocal nInside

        for i in range(j*step,j*step+step):
            x = XrandCoords[i]
            y = YrandCoords[i]        
            # Increment the counter for number of total pins dropped
            
            nDrops = nDrops + 1
            # Check if the points are inside the circle or not
            if x**2+y**2<=radius**2:
                nInside = nInside + 1
                insideX.append(x)
                insideY.append(y)                    
            else:
                outsideX.append(x)
                outsideY.append(y)   
    
            area = 4*nInside/nDrops       
        
        
        return insideX,insideY,outsideX,outsideY,area,nDrops                   

    def update(idx):
        nonlocal nDrops_arr
        nonlocal piValueI
        
        insideX,insideY,outsideX,outsideY,area,nDrops=pi_cal(idx)      
        # print(insideX,insideY,outsideX,outsideY,area,nDrops)
        nDrops_arr.append(nDrops)
        piValueI.append(area) 

        ax_lines.figure.canvas.draw()
        ax_scatter.figure.canvas.draw()
        line_scatter[0].set_data(nDrops_arr,piValueI)        
        xy_scatter=np.array([insideX+outsideX,insideY+outsideY]).T
        line_scatter[1].set_offsets(xy_scatter)
        colors=['k' if i >len(insideX) else'r' for i in range(xy_scatter.shape[0])]
        line_scatter[1].set_facecolors(colors)
        text_scatter.set_text(f'drops={nDrops}; inside_circle={len(insideX)};  pi={area:.6f}')

        return line_scatter

    ani_lines=animation.FuncAnimation(fig,update,frames=tqdm(range(nTrials//step),initial=1, position=0),interval=interval,blit=True) 
    return ani_lines
    
    


if __name__=="__main__":
    # pi_using_MonteCarlo()
    anim=pi_using_MonteCarlo_anim(nTrials=int(5E4))  
    # anim.save('../imgs/3_9_c/pi_mc.gif')
    #HTML(anim.to_jshtml())