# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 18:45:00 2023

@author: richie bao
"""
import libpysal.weights as LW
import esda

def moran_local_autocorrelation_gdf(gdf,col,sw='queen'):
    '''
    局部空间自相关（Local Autocorrelation）——热点（Hot Spots）、冷点（Cold Spots）和空间异常值（Spatial Outliers）计算

    Parameters
    ----------
    gdf : GeoDataFrame
        数据.
    col : str
        待计算列名.
    sw : str, optional
        空间权重，含queen和rookl. The default is 'queen'.

    Returns
    -------
    la_gdf : GeoDataFrame
        局部空间自相关计算结果值.

    '''
    
    if sw=='queen':
        spatial_weight=LW.Queen.from_dataframe(gdf)
    elif sw=='rook':
        spatial_weight=LW.Rook.from_dataframe(gdf)
        
    val=gdf[col]
    li=esda.moran.Moran_Local(val,spatial_weight)    
    
    print(li.q) # values indicate quandrant location 1 HH,  2 LH,  3 LL,  4 HL
    print(li.Is)
    print(f"p_value<0.05 num: {(li.p_sim<0.05).sum()}")    
    
    la_gdf=gdf[['geometry',col]]
    la_gdf["li"]=li.q
    la_gdf["p_value_li"]=li.p_sim
    la_gdf["li_005"]=la_gdf.apply(lambda row:row.li if row.p_value_li<0.05 else 0,axis=1)
    spot_labels=[ '0 ns', '1 hot spot', '2 doughnut', '3 cold spot', '4 diamond']
    la_gdf['cl_li']=la_gdf.apply(lambda row:spot_labels[row.li_005],axis=1)   
    
    return la_gdf
