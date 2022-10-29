# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:32:08 2022

@author: richie bao
"""

def demo_con_style(a_coordi,b_coordi,ax,connectionstyle):
    '''
    function - 在matplotlib的子图中绘制连接线。参考： matplotlib官网Connectionstyle Demo
   
    Params:
        a_coordi - a点的x，y坐标；tuple
        b_coordi - b点的x，y坐标；tuple
        ax - 子图；ax(plot)
        connectionstyle - 连接线的形式；string
        
    Returns:
        None
    '''
    x1, y1=a_coordi[0],a_coordi[1]
    x2, y2=b_coordi[0],b_coordi[1]

    ax.plot([x1, x2], [y1, y2], ".")
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle=connectionstyle,
                                ),
                )

    ax.text(.05, .95, connectionstyle.replace(",", ",\n"),
            transform=ax.transAxes, ha="left", va="top")
    
def demo_con_style_multiple(a_coordi,b_coordi,ax,connectionstyle):
    '''
    function - 在matplotlib的子图中绘制多个连接线
    reference：matplotlib官网Connectionstyle Demo :https://matplotlib.org/3.3.2/gallery/userdemo/connectionstyle_demo.html#sphx-glr-gallery-userdemo-connectionstyle-demo-py

    Params:
        a_coordi - 起始点的x，y坐标；tuple
        b_coordi - 结束点的x，y坐标；tuple
        ax - 子图；ax(plot)
        connectionstyle - 连接线的形式；string
    
    Returns:
        None
    '''
    x1, y1=a_coordi[0],a_coordi[1]
    x2, y2=b_coordi[0],b_coordi[1]

    ax.plot([x1, x2], [y1, y2], ".")
    for i in range(len(x1)):
        ax.annotate("",
                    xy=(x1[i], y1[i]), xycoords='data',
                    xytext=(x2[i], y2[i]), textcoords='data',
                    arrowprops=dict(arrowstyle="<-", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle=connectionstyle,
                                    ),
                    )