# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:44:29 2023

@author: richie bao
"""
import numpy as np
from shapely.geometry import Polygon,Point
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
    
def random_pts_in_bounds(polygon_gdf, number):  
    '''
    根据给定的Polygon格式边界，生成指定数量的随机点坐标值

    Parameters
    ----------
    polygon_gdf : GeoDataFrame
        Polygon边界.
    number : int
        生成随机点坐标值数量.

    Returns
    -------
    x : array(float)
        X坐标值.
    y :  array(float)
        Y坐标值.

    '''    
    
    minx, miny, maxx, maxy=polygon_gdf.bounds.values[0]
    x=np.random.uniform( minx, maxx, number )
    y=np.random.uniform( miny, maxy, number )
    
    return x, y

def random_pts_in_geoBounds(polygon_gdf,number,plot=False):
    '''
    根据给定的Polygon格式边界，生成指定数量对应到Polygon对象投影的GeoDataFrame格式随机点，配合random_pts_in_bounds()函数

    Parameters
    ----------
    polygon_gdf : GeoDataFrame
        Polygon边界.
    number : int
        生成随机点的数量.
    plot : bool, optional
        是否打印边界和随机点地图查看. The default is False.

    Returns
    -------
    pts_in_polygon_gdf : GeoDataFrame
        随机采样点.

    '''    
    
    x,y=random_pts_in_bounds(polygon_gdf, number)
    pts_df=pd.DataFrame()
    pts_df['points']=list(zip(x,y))    
    pts_df['points']=pts_df['points'].apply(Point)
    points_gdf=gpd.GeoDataFrame(pts_df, geometry='points')
    points_gdf.set_crs(polygon_gdf.crs,inplace=True)

    Sjoin=gpd.tools.sjoin(points_gdf,polygon_gdf,predicate="within", how='left')
    idx_name=polygon_gdf.index.values[0]
    pts_in_polygon_gdf=points_gdf[Sjoin.index_right==idx_name]
    
    if plot:
        ax=polygon_gdf.boundary.plot(linewidth=1, edgecolor="black")
        pts_in_polygon_gdf.plot(ax=ax, linewidth=1, color="red", markersize=8)
        plt.show()
    
    return pts_in_polygon_gdf 

def meshgrid_pts_in_geoBounds(polygon_gdf,x_dis=100,y_dis=100):
    '''
    根据给定的Polygon格式边界，生成指定数量对应到Polygon对象投影的GeoDataFrame格式网格点

    Parameters
    ----------
    polygon_gdf : GeoDataFrame
        Polygon边界.
    x_dis : float, optional
        网格单元宽. The default is 100.
    y_dis : float, optional
        网格单元高. The default is 100.

    Returns
    -------
    pts_in_polygon_gdf : GeoDataFrame
        网格采样点.

    '''    
    
    minx, miny, maxx, maxy=polygon_gdf.bounds.values[0]
    nx=np.arange(minx,maxx,x_dis)
    ny=np.arange(miny,maxy,y_dis)
    xv, yv=np.meshgrid(nx, ny)

    pts_df=pd.DataFrame(np.stack([xv,yv],axis=-1).reshape(-1,2), columns = ['x','y'])
    pts_df['points']=pts_df.apply(lambda row:Point(row[0],row[1]),axis=1)

    points_gdf=gpd.GeoDataFrame(pts_df, geometry='points')
    points_gdf.set_crs(polygon_gdf.crs,inplace=True)
    
    Sjoin=gpd.tools.sjoin(points_gdf,polygon_gdf,predicate="within", how='left')
    idx_name=polygon_gdf.index.values[0]
    pts_in_polygon_gdf=points_gdf[Sjoin.index_right==idx_name]
    
    return pts_in_polygon_gdf 
    
def extract_raster_vals_at_pts(pt_coords,raster_fn):
    '''
    对一个栅格数据执行给定采样点提取栅格值

    Parameters
    ----------
    pt_coords : list(float)
        采样点坐标值列表，格式为[(x1,y1),(x2,y2)].
    raster_fn : string
        栅格文件路径名.

    Returns
    -------
    sample_vals : array
        采样点位置栅格值.

    '''    
    import rasterio as rio
    import numpy as np
    
    raster=rio.open(raster_fn)
    sample_vals=np.stack(list(raster.sample(pt_coords)))
    
    return sample_vals

def extract_raster_vals_at_pts_batch(pt_coords,raster_fns_lst):
    '''
    对多个栅格数据执行给定采样点提取栅格值

    Parameters
    ----------
    pt_coords : list(float)
        采样点坐标值列表，格式为[(x1,y1),(x2,y2)].
    raster_fns_lst : list(string)
        多个栅格文件路径名列表.

    Returns
    -------
    sample_vals_array : array
        多个栅格采样点值数组.
    idx_lst : list(int)
        栅格数索引值列表.

    '''    
    from tqdm.notebook import tqdm
    
    idx_lst=[]
    sample_vals_lst=[]
    pbar=tqdm(total=len(raster_fns_lst))
    for i,fn in tqdm(enumerate(raster_fns_lst)):
        sample_vals=extract_raster_vals_at_pts(pt_coords,fn)
        idx_lst.append(i)
        sample_vals_lst.append(sample_vals)
        pbar.update(1)
        
    pbar.close()    
    sample_vals_array=np.concatenate(sample_vals_lst, axis=1)
    return sample_vals_array,idx_lst   
            
