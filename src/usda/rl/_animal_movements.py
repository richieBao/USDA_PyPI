# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:07:04 2023

@author: richie bao
"""
from multiprocessing import Pool
import multiprocessing 
from functools import partial
import numpy as np
from tqdm import tqdm

if __package__:
    from ._animal_movements_pool import neighbors_rewards
else:
    from _animal_movements_pool import neighbors_rewards

def predicting_kde(model,array2d,r=450,ratio_cpu=0.5,interval=1):
    height,width=array2d.shape
    idx_row=np.arange(r, height-r)
    idx_col=np.arange(r,width-r)
    colv, rowv = np.meshgrid(idx_col,idx_row)   
    row_col_v=np.stack((rowv,colv),axis=2)
    row_col_v=row_col_v[::interval,::interval]
    print(row_col_v.shape)
    row_col_v=row_col_v.reshape(-1,2)
    # row_col_v=np.stack((rowv,colv),axis=2).reshape(-1,2)          
    # row_col_v=row_col_v[::interval,:]
    
    cpus = multiprocessing.cpu_count()
    cpus_used=int(cpus*ratio_cpu)
    #print(cpus_used)
    args=partial(neighbors_rewards, args=[model,array2d,r])
    with Pool(cpus_used) as p:
        data_target=p.map(args, tqdm(row_col_v))

    return data_target

if __name__=="__main__":
    import pickle
    import rasterio as rst
    import geopandas as gpd
    
    model_fn=r'C:\Users\richie\omen_richiebao\omen_github\USDA_special_study\models\procyon_lotor_LinearRegression_450.pickle'
    with open(model_fn,'rb') as f:
        model=pickle.load(f)

    procyon_lotor_prj_fn=r'D:\data_B\movebank\procyon_lotor_prj.gpkg'
    procyon_lotor_prj_gdf=gpd.read_file(procyon_lotor_prj_fn)
    
    LC4procyon_lotor_fn=r'C:\Users\richie\omen_richiebao\omen_data_temp\st_louis_1m.tif'
    LC4procyon_lotor_rst=rst.open(LC4procyon_lotor_fn)
    procyon_lotor_prj_vals=LC4procyon_lotor_rst.read(1)    
    
    predicted_kde=predicting_kde(model,procyon_lotor_prj_vals,r=450,ratio_cpu=0.8,interval=100)
    
    # b=set()
    # c=[]
    # for i in dd:        
    #     if i not in b:
    #         c.append(i)
    #     b.add(i)
    
    predicted_kde_fn=r'C:\Users\richie\omen_richiebao\omen_github\USDA_special_study\data\predicted_kde.pickle'    
    with open(predicted_kde_fn,'wb') as f:
        pickle.dump(predicted_kde,f)