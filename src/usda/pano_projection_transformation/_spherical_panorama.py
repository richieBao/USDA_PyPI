# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:29:45 2023

@author: richie bao
"""
import math
import numpy as np
from mayavi import mlab
from tvtk.api import tvtk # python wrappers for the C++ vtk ecosystem

def sphere_panorama_label(image_file,save_fn=None,img_size=(50,50)):
    '''
    转换等量矩形投影图到球面全景

    Parameters
    ----------
    image_file : string
        单张全景图文件路径.

    Returns
    -------
    None.

    '''    
    # create a figure window (and scene)
    fig = mlab.figure(size=(600, 600),bgcolor=(1, 1, 1))

    # load and map the texture
    img = tvtk.JPEGReader()
    img.file_name = image_file
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)
    # print(texture)
    # (interpolate for a less raster appearance when zoomed in)

    # use a TexturedSphereSource, a.k.a. getting our hands dirty
    R = 50
    Nrad = 180

    # create the sphere source with a given radius and angular resolution
    sphere = tvtk.TexturedSphereSource(radius=R, theta_resolution=Nrad,phi_resolution=Nrad)

    # assemble rest of the pipeline, assign texture    
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
    fig.scene.add_actor(sphere_actor)
    
    # Plot the equator and the tropiques
    theta_equator=np.linspace(0, 2 * np.pi, 100)
    veiw_scope_dic={}
    for i,angle in enumerate([-math.radians(70), 0, math.radians(50)]):
        x_equator=R * np.cos(theta_equator) * np.cos(angle)
        y_equator=R * np.sin(theta_equator) * np.cos(angle)
        z_equator=R * np.ones_like(theta_equator) * np.sin(angle)    
        mlab.plot3d(x_equator, y_equator, z_equator, color=(0, 0, 0),opacity=0.6, tube_radius=None)  
        veiw_scope_dic[i]=[x_equator,y_equator,z_equator]    
    
    str_info={0:'lower limit of visual filed:-70',1:'Standard line of sight:0',2:'Upper limit of visual filed:+50'}
    for k,v in str_info.items():
        mlab.text(veiw_scope_dic[k][0][0], veiw_scope_dic[k][1][0], v, z=veiw_scope_dic[k][2][0],width=0.029 * len(v), name=v,color=(0,0,0))
    
    vertical_label_radians=np.linspace(0, np.pi,14)
    vertical_label_degree=["{:.2f}".format(90-math.degrees(radi)) for radi in vertical_label_radians]
    phi_label=0
    for idx in range(len(vertical_label_radians)):
        theta_labe=vertical_label_radians[idx]
        x_label=R * np.sin(theta_labe) * np.cos(phi_label)
        y_label=R * np.sin(theta_labe) * np.sin(phi_label)
        z_label=R * np.cos(theta_labe)         
        mlab.points3d(x_label, y_label, z_label,scale_factor=1,color=(0,0,0))
        label=vertical_label_degree[idx]
        mlab.text(x_label, y_label, label, z=z_label,width=0.028 * len(label), name=label,color=(0,0,0))    
        
    if save_fn:
        mlab.savefig(save_fn,size=img_size)
        
    mlab.show()

if __name__ == "__main__":
    # # from database import cfg_load_yaml 
    # cfg=cfg_load_yaml('config.yml')
    
    #panorama_example_fn=cfg["spherical_panorama"]["panorama_example_fn"]
    #sphere_panorama_label(panorama_example_fn)  
    # panorama_seg_example_fn=cfg["spherical_panorama"]["panorama_seg_example_fn"]
    # sphere_panorama_label(panorama_seg_example_fn)  
    
    fn=r'G:\data\pano_dongxistreet\pano_projection_transforms\img_seg_redefined_color\dongxistreet_228.jpg'
    sphere_panorama_label(fn)
    