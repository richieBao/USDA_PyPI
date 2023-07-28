# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 10:30:28 2023

@author: richie bao
"""
from PIL import Image,ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib

import os
from mpl_toolkits.axes_grid1 import ImageGrid

from matplotlib.patches import Arrow, Circle,Patch
from tqdm import tqdm  
import pandas as pd

from sklearn.cluster import KMeans
from collections import Counter
from pathlib import Path
from skimage import measure

if __package__:    
    from ._color_metrics_pool import get_image
    from ._color_metrics_pool import RGB2HEX
else:
    from _color_metrics_pool import get_image
    from _color_metrics_pool import RGB2HEX

    
def panoramic_transform_example_show(*args):
    '''
    全景图投影变换示例排布绘图

    Parameters
    ----------
    *args : 图像路径参数
        包括：panorama_fn,polar_fn,cube_fn,sphere_fn.

    Returns
    -------
    None.

    '''    
    panorama_fn,polar_fn,cube_fn,sphere_fn=args
    
    font = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 28}
    matplotlib.rc('font', **font)
    title_fontsize=55
    
    # vertical_label_radians=np.linspace(0, np.pi,14)
    # vertical_label_degree=["{:.2f}".format(90-math.degrees(radi)) for radi in vertical_label_radians]
    vertical_label_degree=90-np.linspace(0, 180,17)
    horizontal_label_degree=180-np.linspace(0, 360,17)    
    print(horizontal_label_degree)
    
    pano=np.asarray(Image.open(panorama_fn))
    # pano=cv2.imread(panorama_fn)
    pano_height,pano_width,_=pano.shape
    print(pano_width,pano_height)
    
    # fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=(20,10))
    fig=plt.figure(figsize=(20,10))
    
    #01-pano
    # ax1=plt.subplot(131, frameon=False)  
    ax1_coords=[0, 0, 1, 1] #rect : This parameter is the dimensions [left, bottom, width, height] of the new axes.
    ax1=fig.add_axes(ax1_coords)    
    
    ax1.imshow(pano,)     #extent=[x0, x1, y0, y1]  interpolation='bilinear',aspect='auto',origin='lower',
    ax1.set_yticks(np.linspace(0,pano_height,len(vertical_label_degree)))    
    ax1.set_yticklabels(vertical_label_degree) # provide name to the x axis tick marks   
    ax1.set_xticks(np.linspace(0,pano_width,len(horizontal_label_degree)))    
    ax1.set_xticklabels(horizontal_label_degree) # provide name to the x axis tick marks   
    ax1.axhline(y=pano_height/2,color='gray',linestyle='-.',linewidth=1)
    ax1.axvline(x=pano_width/2,color='gray',linestyle='-.',linewidth=1)
    ax1.set_title("Equirectangular format", va = 'bottom',fontsize=title_fontsize)

    #02-polar
    # polar=cv2.imread(polar_fn)
    polar=np.asarray(Image.open(polar_fn))
    polar_height,polar_width,_=polar.shape   
    ax2_coords = [0.8, 0, 1, 1]
    
    ax2_image = fig.add_axes(ax2_coords)
    ax2_image.imshow(polar, alpha = 1)
    ax2_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image    

    #theta = np.linspace(0, 2 * np.pi, 73)       
    ax2_coords_=[0.80000001, 0, 1, 1] #如果位置重叠，可能会提示ValueError: Unknown element o错误
    ax2_polar = fig.add_axes(ax2_coords_, projection='polar')
    ax2_polar.patch.set_alpha(0)
    ax2_polar.set_ylim(30, 41)
    ax2_polar.set_yticks(np.arange(30, 41, 2))
    ax2_polar.set_yticklabels([])
    ax2_polar.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax2_polar.grid(True)
    ax2_polar.set_title("Polar format", va = 'bottom',fontsize=title_fontsize)    

    #03-cube
    ax3_coords=[1.47, 0, 1, 1]
    ax3=fig.add_axes(ax3_coords)
    # cube=cv2.imread(cube_fn)
    cube=np.asarray(Image.open(cube_fn))
    cube_height,cube_width,_=cube.shape
    ax3.imshow(cube, alpha = 1)
    ax3.axis('off')    
    ax3.axvline(x=cube_width*(1/3),color='gray',linestyle='-.',linewidth=1)
    ax3.axvline(x=cube_width*(2/3),color='gray',linestyle='-.',linewidth=1)    
    ax3.set_title("Cubic format", va = 'bottom',fontsize=title_fontsize)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.4)
    ax3.text(cube_width*(1/6), cube_height*(1/4), "Top", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(3/6), cube_height*(1/4), "Back", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(5/6), cube_height*(1/4), "Down", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(1/6), cube_height*(3/4), "Left", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(3/6), cube_height*(3/4), "Front", ha="center", va="center", size=40,bbox=bbox_props)
    ax3.text(cube_width*(5/6), cube_height*(3/4), "Right", ha="center", va="center", size=40,bbox=bbox_props)    
    
    #04-sphere
    ax4_coords=[2.15, 0, 1, 1]
    ax4=fig.add_axes(ax4_coords)    
    # sphere=cv2.imread(sphere_fn)    
    sphere=np.asarray(Image.open(sphere_fn))
    ax4.imshow(sphere)     
    ax4.axis('off') 
    ax4.set_title("Spherical format", va = 'bottom',fontsize=title_fontsize)
    
    fig.tight_layout()    
    # fig.savefig('./graph/preprocessed data_01',dpi=300)
    plt.show()

def imgs_arranging(imgs_root,img_fns,save_path):
    '''
    语义分割立方体型全景图不同对象视域占比示例

    Parameters
    ----------
    imgs_root : dict
        图像根目录.
    img_fns : dict
        选择不同区间像素占比的图像.
    save_path : string
        保持排布图像的路径.

    Returns
    -------
    None.

    '''    
    font = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 28}
    matplotlib.rc('font', **font)    
    
    rows=['(-0.001, 15.0]','(15.0, 25.0]','(25.0, 50.0]','(50.0, 100.0]']
    img_list=[]
    for k,fns in img_fns.items():
        img_list.append([os.path.join(imgs_root,i[1]+'.jpg') for i in fns])
    img_list=list(map(list,zip(*img_list)))
    
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
    img_list=flatten_lst(img_list)
    nrows_ncols=(len(list(img_fns.values())[0]),len(img_fns.keys()),)
    print(nrows_ncols)
    fig=plt.figure(figsize=(30.+1, 20.-3))
    grid=ImageGrid(fig, 111,  # similar to subplot(111)
                   nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                   axes_pad=0.05,  # pad between axes in inch.
                     )    
    i=0
    for ax,im_fn in zip(grid,img_list):
        # print(ax)
        ax.imshow(Image.open(im_fn))           
        # ax.axis('off')         
        
        if i<5:ax.set_title(list(img_fns.keys())[i])        
        if i%5==0:ax.set_ylabel(rows[i//5], )
        i+=1
    fig.tight_layout()
    # plt.show()
    plt.savefig(save_path,dpi=300)
    

def skyline_metrics_imgs_arranging(idxes_df,imgs_root,save_path,domain,idx='PA'):
    '''
    不同数量级的天际线形状指数的极坐标形式示例

    Parameters
    ----------
    idxes_df : DataFrame
        天际线景观指数.
    imgs_root : string
        极坐标格式全景图根目录.
    save_path : string
        排布图像保存路径.
    domain : dict
        不同指数计算对应提取区间的配置.
    idx : string, optional
        所要计算的指数. The default is 'PA'.

    Returns
    -------
    None.

    '''    
    idx_domain=domain[idx]
    if idx=='PA':
        lower_df=idxes_df[idxes_df.perimeter_area_ratio_mn<idx_domain[1]-20]
        upper_df=idxes_df[idxes_df.perimeter_area_ratio_mn>=idx_domain[1]+140]
        title_n='Perimeter area ratio'
    elif idx=='SI':
        lower_df=idxes_df[idxes_df.shape_index_mn<idx_domain[1]-0.5]
        upper_df=idxes_df[idxes_df.shape_index_mn>=idx_domain[1]+1]
        title_n='Shape index'
    elif idx=='FD':
        lower_df=idxes_df[idxes_df.fractal_dimension_mn<idx_domain[1]-0.01]
        upper_df=idxes_df[idxes_df.fractal_dimension_mn>=idx_domain[1]+0.01]  
        title_n='Fractal dimension'
        
    lower_fns=lower_df.fn_stem.sample(n=10)
    upper_fns=upper_df.fn_stem.sample(n=10)        

    lower_upper_fns=lower_fns.append(upper_fns)
    lower_upper_fns=[os.path.join(imgs_root,i+'.jpg') for i in lower_upper_fns]

    font = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 26}
    matplotlib.rc('font', **font) 
    
    nrows_ncols=(2,10)
    fig=plt.figure(figsize=(30, 7.))
    grid=ImageGrid(fig, 111,  # similar to subplot(111)
                   nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
                   axes_pad=0.0,  # pad between axes in inch.
                     )      
    i=0
    rows=['({},{}]'.format(idx_domain[0],idx_domain[1]),'({},{}]'.format(idx_domain[1],idx_domain[2])]
    for ax,im_fn in zip(grid,lower_upper_fns):
        # print(ax)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_frame_on(False)
        ax.imshow(Image.open(im_fn),cmap='Greys',  interpolation='nearest')  
        if i==1:ax.set_title(title_n)   
        if i%10==0:ax.set_ylabel(rows[i//10], )
        i+=1
        
    # plt.title(title_n)      
    fig.tight_layout()    
    # plt.show()        
    plt.savefig(os.path.join(save_path,'{}.png'.format(title_n)),dpi=300)  

def dominant2cluster_colors_imshow(fn,coords,colors_dominant_entropy,save_path=None,resize_scale=0.1,number_of_colors=16):
    '''
    色彩丰富度指数示例排布

    Parameters
    ----------
    fn : string
        示例图像路径.
    coords : dict
        各个道路对应全景图的采集坐标点.
    save_path : string
        待保持图像的根目录.
    colors_dominant_entropy: df
        色彩丰富度（均衡都）指数
    resize_scale : numerical val, optional
        调整图像大小的比例. The default is 0.1.
    number_of_colors : int, optional
        主题色提取的数量. The default is 16.

    Returns
    -------
    None.

    '''    
    font = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)      
        
    fig, axs=plt.subplots(1, 4, figsize=(30, 8))
    
    fn_stem=Path(fn).stem
    fn_key,fn_idx=fn_stem.split("_")
    
    img=get_image(fn)
    img=img[:int(img.shape[0]*(70/100))]
    axs[0].imshow(img)
    axs[0].set_title('Panorama') 
    axs[0].set_ylabel('color richness index:{}'.format(round(colors_dominant_entropy.loc[colors_dominant_entropy['fn_stem']==os.path.basename(fn).split('.')[0]].counter.values.item(),3))) #64.178

    img_h,img_w,_=img.shape
    modified_img_w,modified_img_h=int(img_w*resize_scale),int(img_h*resize_scale),
    modified_img=cv2.resize(img, (modified_img_w,modified_img_h), interpolation = cv2.INTER_AREA)
    modified_img=modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    clf=KMeans(n_clusters=number_of_colors)
    labels=clf.fit_predict(modified_img)
    center_colors=clf.cluster_centers_
    
    labels_RGB=np.array([center_colors[i] for i in labels])
    labels_RGB_restore=labels_RGB.reshape((modified_img_h,modified_img_w,3))
    axs[1].imshow(labels_RGB_restore/255)
    axs[1].set_title('Theme color distribution') 
    
    labels_restored=labels.reshape((modified_img_h,modified_img_w,))    
    img_labeled=measure.label(labels_restored, connectivity=1)
    axs[3].imshow(img_labeled,cmap="gist_ncar")
    axs[3].set_title('Theme color proximity clustering')
    
    counts=Counter(labels)    
    ordered_colors=[center_colors[i] for i in counts.keys()] # We get ordered colors by iterating through the keys
    hex_colors=[RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors=[ordered_colors[i] for i in counts.keys()]

    axs[2].pie(counts.values(), labels=hex_colors, colors=hex_colors,rotatelabels =True,radius=0.5)   #labels=hex_colors,
    axs[2].set_title('Theme color') 
    
    fig.tight_layout() 
    # plt.show()
    if save_path:
        plt.savefig(os.path.join(save_path,os.path.basename(fn)),dpi=300) 
    else:
        plt.show()

def kp_show(feature_map,imgs_path_list,imgs_root,save_path=None):
    '''
    特征点邻域大小比较示例排布

    Parameters
    ----------
    feature_map : list
        图像的特征映射（码本映射）.
    imgs_path_list : list
        示例文件名列表.
    imgs_root : string
        图像根目录.
    save_path : string
        图像保存文件名.

    Returns
    -------
    None.

    '''
    font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 28}
    matplotlib.rc('font', **font)
    
    fig, axs=plt.subplots(1, 2, constrained_layout=True,figsize=(50,10))
    axs=axs.flat
    bins=[0,10,20,30,40]
    for idx,fn_stem in enumerate(imgs_path_list):        
        img=get_image(os.path.join(imgs_root,fn_stem+'.jpg'))
        # print(img.shape)
        img=img[:int(img.shape[0]*(70/100))]    
        # plt.imshow(img)   
        kp_dict={i['fn_stem']:i['kp'] for i in feature_map}
        kp=kp_dict[fn_stem]
        # print(kp)
        kp_df=pd.DataFrame.from_dict(kp)
        # print(kp_df)
        fre_size=kp_df[['size']].apply(pd.Series.value_counts,bins=bins,).to_dict()['size']
        fre_size={'{}-{}'.format(k.left,k.right):v for k,v in fre_size.items()}
        # print(fre_size)
        patches_1=[Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=3, color='red',fill=True,alpha=0.8) if kp[i]['size'] in range(0,10) 
                 else Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=3, color='b',fill=True,alpha=0.8) for i in range(len(kp))]
        patches_2=[Circle((kp[i]['pt'][0], kp[i]['pt'][1]),radius=kp[i]['size'], color='red',fill=True,alpha=0.8) if kp[i]['size'] in range(0,10) 
                 else Circle((kp[i]['pt'][0], kp[i]['pt'][1]), radius=kp[i]['size'], color='b',fill=False,alpha=0.8) for i in range(len(kp))]        
        
        axs[idx].imshow(img)
        for p in patches_2:
            axs[idx].add_patch(p) 
        axs[idx].set_title('key-points size:{}'.format(fre_size))
    C_1=Patch(color='red',fill=True,alpha=0.8,label='kp size in (0,10]')
    C_2=Patch(color='b',fill=False,alpha=0.8,label='kp size not in (0,10]')        
    plt.legend(handles=[C_1,C_2])    
    # plt.show()  
    if save_path:
        plt.savefig(save_path,dpi=300)        
    else:
        plt.show()  

# if __name__=="__main__":
#     from database import postSQL2gpd,gpd2postSQL,cfg_load_yaml      
#     import os    
#     import pickle
#     cfg=cfg_load_yaml('config.yml')       
#     UN=cfg["postgreSQL"]["myusername"]
#     PW=cfg["postgreSQL"]["mypassword"]
#     DB=cfg["postgreSQL"]["mydatabase"]    
#     GC='geometry'  
    
#     #A.全景图投影变换示例排布绘图
#     panoramic_transform_example_fn=cfg["imgs_arranging"]["panoramic_transform_example_fn"]
#     PT_img_fns=[os.path.join(cfg["streetview"]["panoramic_imgs_valid_root"],panoramic_transform_example_fn),
#                 os.path.join(cfg["equi2polar"]["region"]["polar_img_root"],panoramic_transform_example_fn),
#                 os.path.join(cfg["equi2cube"]["cube_img"],panoramic_transform_example_fn),
#                 os.path.join(cfg["spherical_panorama"]["img_sphere_root"],panoramic_transform_example_fn)]
#     panoramic_transform_example_show(*PT_img_fns)
    
#     PT_seg_fns=[os.path.join(cfg["panaSeg"]["pano_path"],panoramic_transform_example_fn),
#                 os.path.join(cfg["equi2polar"]["region"]["polar_seg_root"],panoramic_transform_example_fn),
#                 os.path.join(cfg["equi2cube"]["region"]["img_seg_cube_root"],panoramic_transform_example_fn),
#                 os.path.join(cfg["spherical_panorama"]["seg_sphere_root"],panoramic_transform_example_fn)]
    # panoramic_transform_example_show(*PT_seg_fns)
    
    #B.语义分割立方体型全景图不同对象视域占比示例
    # cube_object_percentage_example_fn=cfg['imgs_arranging']['cube_object_percentage_example_fn']
    # object_percentage_cube_save_path=cfg['imgs_arranging']['object_percentage_cube_save_path']
    # cube_imgs_root=cfg['equi2cube']['cube_img']
    # imgs_arranging(cube_imgs_root,cube_object_percentage_example_fn,object_percentage_cube_save_path)
    
    #C.不同数量级的天际线形状指数的极坐标形式示例
    # GC='geometry'  
    # metrics_skyline_shape_gdf=postSQL2gpd(table_name='metrics_skyline_shape',geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
    # skyline_metrics_domain=cfg['imgs_arranging']['skyline_metrics_domain']
    # polar_seg_root=cfg['equi2polar']['region']['polar_sky_root'] 
    # skyline_metrics_imgs_save_path=cfg['imgs_arranging']['skyline_metrics_imgs_save_path']
    # for idx in skyline_metrics_domain.keys():
    #     skyline_metrics_imgs_arranging(metrics_skyline_shape_gdf,polar_seg_root,skyline_metrics_imgs_save_path,skyline_metrics_domain,idx=idx)

    #D.色彩丰富度指数示例
    # from analysis_n_results.color_metrics_pool import get_image,RGB2HEX     
    # with open(cfg['streetview']['save_path_BSV_retrival_info']['coords'],'rb') as f: 
    #     coords=pickle.load(f)    
    # theme_color_cluster_save_path=cfg['imgs_arranging']['theme_color_cluster_save_path']
    # resize_scale_cluster=cfg['color_metrics']['resize_scale_cluster']
    # number_of_colors=cfg['color_metrics']['number_of_colors']
    # img_path=cfg['streetview']['panoramic_imgs_valid_root']
    # theme_color_cluster_example_fn=cfg['imgs_arranging']['theme_color_cluster_example_fn']
    # colors_dominant_entropy_table_name=cfg['color_metrics']['colors_dominant_entropy_table_name']
    # colors_dominant_entropy_gdf=postSQL2gpd(table_name=colors_dominant_entropy_table_name,geom_col=GC,myusername=UN,mypassword=PW,mydatabase=DB)
    # for img_fn in theme_color_cluster_example_fn:        
    #     dominant2cluster_colors_imshow(os.path.join(img_path,img_fn),coords,theme_color_cluster_save_path,colors_dominant_entropy_gdf,resize_scale_cluster,number_of_colors)
    
    #E.特征点邻域大小比较示例
    # img_path=cfg['streetview']['panoramic_imgs_valid_root']
    # feature_map_region_fn=cfg['KP_metrics']['feature_map_region_fn']
    # with open(feature_map_region_fn,'rb') as f:
    #     feature_map_region=pickle.load(f)     
    # kp_stats_example_fn=cfg['imgs_arranging']['kp_stats_example_fn']
    # kp_stats_example_save_path=cfg['imgs_arranging']['kp_stats_example_save_path']
    # kp_show(feature_map_region,kp_stats_example_fn,img_path,kp_stats_example_save_path) 