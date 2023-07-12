# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:22:49 2023

@author: richie bao
"""
# import rioxarray as rxr
# from sklearn.utils import Bunch
# import os
import uuid
import numpy as np
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

def data_target_func(df,args):
    raster,y_columns,x_size,y_size,x_offset,y_offset=args   
    # print(y_columns)
    data = []
    target = []
    for idx,row in df.iterrows():
        try:            
            clipped_raster=raster.rio.clip([row.geometry],from_disk=True)            
            y=row[y_columns]
            clipped_raster_adj=clipped_raster.data[:,x_offset:x_offset+x_size,y_offset:y_offset+y_size] 
            if clipped_raster_adj.shape==(1,x_size,y_size): 
                data.append(clipped_raster_adj)
                target.append(y)   
        except:
            pass
    if len(data)>0:    
        data = np.array(data)
        target = np.array(target)

        return [data,target]
    
        
if __name__=="__main__":
    with open(r'I:\data\london\lc_lst_dataset\lcNlst_e261062e-eb93-4bfa-8c13-6d5029c7d4e4.pickle','rb') as f:
        a=pickle.load(f)
