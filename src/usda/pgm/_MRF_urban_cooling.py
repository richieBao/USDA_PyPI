# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:03:51 2023

@author: richie bao
"""
import warnings
warnings.filterwarnings('ignore')

import os
import rioxarray as rxr
import numpy as np
import geopandas as gpd

class MRF_urban_cooling:

    def __init__(self, args, Urban_cooling, delta_tmp=-2,epochs=1,alpha=1,beta=1):        
        self.alpha=alpha
        self.beta=beta
        self.Urban_cooling=Urban_cooling
      
        lulc=rxr.open_rasterio(args["lulc_raster_path"],masked=True).squeeze()  
        lulc_array=lulc.values     
        self.X=np.copy(lulc_array)

        self.X_rio=lulc.copy()
        self.X_rio.values=self.X
        self.X_rio_fn=os.path.join(args['workspace_dir'],'X.tif')
        self.X_rio.rio.to_raster(self.X_rio_fn)      
        self.X_rio.close()

        self.Urban_cooling(args)
        uhi_results=gpd.read_file(os.path.join(args['workspace_dir'],'uhi_results.shp'))
        self.avg_tmp=uhi_results['avg_tmp_v'].values[0]
        self.objective_tmp=delta_tmp+self.avg_tmp
        print(f'current temperature={self.avg_tmp:.3f}\nobjective temperature={self.objective_tmp:.3f}\nlables={np.unique(lulc_array)}\nsize={lulc_array.shape}')
        print('-'*50)

        self.args=args
        self.args.update({"lulc_raster_path":self.X_rio_fn})
        self.epochs=epochs 
        self.X_stack=[]

        self.tmp_gap_val_a=abs(delta_tmp)
        self.updated_ij_num=0
        self.tmp_gap_val_lst={}
        
        for it in range(epochs):            
            self.MRF(lulc_array)
            self.X_stack.append(self.X)
            tmp_gap_val_epoch=self.tmp_gap()            
            print(f'\nepoch={it};temperatur gap={tmp_gap_val_epoch}')            

    def MRF(self,obs):
        (M,N)=obs.shape
        labels=np.unique(obs)        
        
        for i in range(M):
            for j in range(N):
                X_ij=self.X[i,j]
                cost=[self.energy(obs,lb,i,j) for lb in labels]
                min_val=min(cost)
                min_idxes=[i for i, v in enumerate(cost) if v == min_val]                
                min_labels=[labels[i] for i in min_idxes]
                current_label=obs[i,j] # ? obs or X_temp

                self.X[i,j]=X_ij 
                if current_label not in min_labels:   
                    self.X[i,j]=min_labels[0]  
                    self.updated_X_to_raster()
                    
                    tmp_gap_val_b=self.tmp_gap() 
                    if tmp_gap_val_b<self.tmp_gap_val_a:   
                        self.tmp_gap_val_a=tmp_gap_val_b
                        self.tmp_gap_val_lst[(i,j)]=tmp_gap_val_b
                        self.updated_ij_num+=1
                        print(f'cell=({i},{j});current label={current_label};target label={min_labels[0]};updated_ij num={self.updated_ij_num};temperatur gap={self.tmp_gap_val_a}') # ,end='\r',flush=True) 
                    else:
                        self.X[i,j]=X_ij 
                        self.updated_X_to_raster()                        
  
                print(f'cell=({i},{j});current label={current_label};target label={current_label};updated_ij num={self.updated_ij_num};temperatur gap={self.tmp_gap_val_a}') # ,end='\r',flush=True) 

    def energy(self,obs,lb,i,j):
        edge_potential=self.clique(i,j,lb,self.X)  
        
        self.X[i,j]=lb
        self.updated_X_to_raster()
        # diff_lulc_sum=(self.X==obs).sum()/obs.size
        tmp_gap_val=self.tmp_gap()

        return self.beta*tmp_gap_val+self.alpha*edge_potential 

    def updated_X_to_raster(self):
        self.X_rio.values=self.X
        self.X_rio.rio.to_raster(self.X_rio_fn) 
        self.X_rio.close() 

    def tmp_gap(self):        
        self.Urban_cooling(self.args)
        uhi_results=gpd.read_file(os.path.join(self.args['workspace_dir'],'uhi_results.shp'))
        avg_tmp_X=uhi_results['avg_tmp_v'].values[0]        
        tmp_gap=abs(self.objective_tmp-avg_tmp_X)       

        return tmp_gap

    def clique(self,i,j,lb,X):
        (M,N)=X.shape        
        # find correct neighbors
        if (i==0 and j==0):
            neighbor=[(0,1),(1,0)]
        elif i==0 and j==N-1:
            neighbor=[(0,N-2),(1,N-1)]
        elif i==M-1 and j==0:
            neighbor=[(M-1,1),(M-2,0)]
        elif i==M-1 and j==N-1:
            neighbor=[(M-1,N-2),(M-2,N-1)]
        elif i==0:
            neighbor=[(0,j-1),(0,j+1),(1,j)]
        elif i==M-1:
            neighbor=[(M-1,j-1),(M-1,j+1),(M-2,j)]
        elif j==0:
            neighbor=[(i-1,0),(i+1,0),(i,1)]
        elif j==N-1:
            neighbor=[(i-1,N-1),(i+1,N-1),(i,N-2)]
        else:
            neighbor=[(i-1,j),(i+1,j),(i,j-1),(i,j+1),(i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]
            
        return sum(self.delta(lb,X[i]) for i in neighbor)
        
    def delta(self,a,b):
        if a==b:
            return -1
        else:
            return 1
       
if __name__=="__main__":        
    from usda.migrated_project.invest.esv import Urban_cooling
    
    data_root=r'I:\data\invest\UrbanCoolingModel'
    args={
        "aoi_vector_path": os.path.join(data_root,"aoi.shp"), 
        "avg_rel_humidity": 30, 
        "biophysical_table_path": os.path.join(data_root,"Biophysical_UHI_fake.csv"), 
        # "building_vector_path": os.path.join(data_root,"sample_buildings.shp"), 
        "cc_method": "factors",
        "cc_weight_albedo": 0.2, 
        "cc_weight_eti": 0.2, 
        "cc_weight_shade": 0.6, 
        "do_energy_valuation": False, 
        "do_productivity_valuation": False, 
        "energy_consumption_table_path": os.path.join(data_root,"Fake_energy_savings.csv"), 
        "green_area_cooling_distance": 1000, 
        "lulc_raster_path": os.path.join(data_root,"lulc.tif"), 
        "ref_eto_raster_path": os.path.join(data_root,"et0.tif"), 
        # "results_suffix": "cooling", 
        "t_air_average_radius": 2000, 
        "t_ref": 21.5, 
        "uhi_max": 3.5,
        "workspace_dir":r'I:\ESVs\urban_cooling',
    }    
        
    muc=MRF_urban_cooling(args,Urban_cooling,epochs=1,alpha=1,beta=1) 
    
    import pickle
    with open(os.path.join(args['workspace_dir'],'tmp_gap_val.pkl'),'wb') as f:
        pickle.dump(muc.tmp_gap_val_lst,f)