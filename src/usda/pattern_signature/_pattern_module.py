# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:25:53 2023

@author: richie bao
"""
from ..pattern_signature import  _signature as usda_signature
from ..utils import _df_process as usda_df_process
from ..pattern_signature import _distance_metric as usda_distance

import cc3d
import numpy as np
import rioxarray as rxr
from tqdm import tqdm
from shapely.geometry import mapping

def pattern_search(query_scene_array,quadrats_gdf,tif_fn,signature='class_clumpSize_histogram',distance_metirc='Jensen-Shan'):  
    '''
    给定一个2d数组（栅格数据），指定signature和distance的方法，返回distance值

    Parameters
    ----------
    query_scene_array : 2darray
        2维数组， 参照栅格数据.
    quadrats_gdf : GeoDataFrame
        为比对的样方Polygon对象.
    tif_fn : str
        栅格数据文件路径.
    signature : str, optional
        signature标记方法名称. The default is 'class_clumpSize_histogram'.
    distance_metirc : str, optional
        distance距离测量方法名称. The default is 'Jensen-Shan'.

    Returns
    -------
    quadrats_gdf_copy : GeoDataFrame
        含distance结果的数据.

    '''
    signature_func_dict={'class_clumpSize_histogram':lambda v: usda_signature.class_clumpSize_histogram(v,cc3d.connected_components(v,connectivity=8,return_N=False,out_dtype=np.uint64)),
                         'class_pairs_frequency':lambda v: usda_signature.class_co_occurrence(v),
                         'class_hierarchical_decomposition':lambda v: usda_signature.class_decomposition(v)}
    
    distance_func_dict={'Jensen-Shan':lambda instance:instance.shannon()['Jensen-Shan'],
                        'Wave Hedges':lambda instance:instance.intersection()['Wave Hedges'],
                        'Jaccard':lambda instance:instance.inner()['Jaccard']} 
    
    sig_query_scene=list(map(signature_func_dict[signature],[query_scene_array]))[0]
    sig_query_scene=sig_query_scene/sig_query_scene.values.sum()
    scenes=rxr.open_rasterio(tif_fn,masked=True).squeeze()    
    distance_dict={}
    for idx,row in tqdm(quadrats_gdf.iterrows(),total=quadrats_gdf.shape[0]):
        scene=scenes.rio.clip([mapping(row.geometry)],quadrats_gdf.crs)
        sig_scene=list(map(signature_func_dict[signature],[scene.data]))[0]
        sig_scene=sig_scene/sig_scene.values.sum()
        
        sig_query_scene,sig_scene=usda_df_process.complete_dataframe_rowcols([sig_query_scene,sig_scene])    
        distance_instance=usda_distance.Distances(sig_query_scene.to_numpy().flatten(),sig_scene.to_numpy().flatten()) 
        distance=list(map(distance_func_dict[distance_metirc],[distance_instance]))[0]
        distance_dict[idx]=distance

    quadrats_gdf_copy=quadrats_gdf.copy(deep=True)
    quadrats_gdf_copy['distance']=distance_dict.values()
    
    return quadrats_gdf_copy

def pattern_compare(quadrats_gdf,tif_a_fn,tif_b_fn,signature='class_clumpSize_histogram',distance_metirc='Jensen-Shan'):
    '''
    给两个栅格数据（2维数据），例如不同年份同一区域的LULC（土地利用/土地覆盖），指定样方，signature和distance方法，比较两个栅格的变化差异

    Parameters
    ----------
    quadrats_gdf :GeoDataFrame
        为比对的样方Polygon对象.
    tif_a_fn : str
        栅格数据1.
    tif_b_fn : str
        栅格数据2.
    signature : str, optional
        signature标记方法名称. The default is 'class_clumpSize_histogram'.
    distance_metirc : str, optional
        distance距离测量方法名称. The default is 'Jensen-Shan'.

    Returns
    -------
    quadrats_gdf_copy : GeoDataFrame
        含distance结果的数据.

    '''
    signature_func_dict={'class_clumpSize_histogram':lambda v: usda_signature.class_clumpSize_histogram(v,cc3d.connected_components(v,connectivity=8,return_N=False,out_dtype=np.uint64)),
                         'class_pairs_frequency':lambda v: usda_signature.class_co_occurrence(v),
                         'class_hierarchical_decomposition':lambda v: usda_signature.class_decomposition(v)}
    
    distance_func_dict={'Jensen-Shan':lambda instance:instance.shannon()['Jensen-Shan'],
                        'Wave Hedges':lambda instance:instance.intersection()['Wave Hedges'],
                        'Jaccard':lambda instance:instance.inner()['Jaccard']} 
    
    scenes_a=rxr.open_rasterio(tif_a_fn,masked=True).squeeze() 
    scenes_b=rxr.open_rasterio(tif_b_fn,masked=True).squeeze() 
    
    distance_dict={}    
    for idx,row in tqdm(quadrats_gdf.iterrows(),total=quadrats_gdf.shape[0]):
        scene_a=scenes_a.rio.clip([mapping(row.geometry)],quadrats_gdf.crs)
        scene_b=scenes_b.rio.clip([mapping(row.geometry)],quadrats_gdf.crs)
        sig_a=list(map(signature_func_dict[signature],[scene_a.data]))[0]
        sig_b=list(map(signature_func_dict[signature],[scene_b.data]))[0]
        sig_a=sig_a/sig_a.values.sum()
        sig_b=sig_b/sig_b.values.sum()
        
        sig_a,sig_b=usda_df_process.complete_dataframe_rowcols([sig_a,sig_b])
        distance_instance=usda_distance.Distances(sig_a.to_numpy().flatten(),sig_b.to_numpy().flatten()) 
        distance=list(map(distance_func_dict[distance_metirc],[distance_instance]))[0]
        distance_dict[idx]=distance

    quadrats_gdf_copy=quadrats_gdf.copy(deep=True)
    quadrats_gdf_copy['distance']=distance_dict.values()
    
    return quadrats_gdf_copy

def pattern_reference_distance(quadrats_gdf,tif_fn,signature='class_clumpSize_histogram',distance_metirc='Jensen-Shan'):
    '''
    与同样方大小，值为0的2维矩阵，给定signature和distance比较距离

    Parameters
    ----------
    quadrats_gdf : GeoDataFrame
        为比对的样方Polygon对象.
    tif_fn : str
        栅格数据文件路径名.
    signature : str, optional
       signature标记方法名称. The default is 'class_clumpSize_histogram'.
    distance_metirc : str, optional
        distance距离测量方法名称. The default is 'Jensen-Shan'.

    Returns
    -------
    quadrats_gdf_copy : GeoDataFrame
        含distance结果的数据.

    '''
    signature_func_dict={'class_clumpSize_histogram':lambda v: usda_signature.class_clumpSize_histogram(v,cc3d.connected_components(v,connectivity=8,return_N=False,out_dtype=np.uint64)),
                         'class_pairs_frequency':lambda v: usda_signature.class_co_occurrence(v),
                         'class_hierarchical_decomposition':lambda v: usda_signature.class_decomposition(v)}
    
    distance_func_dict={'Jensen-Shan':lambda instance:instance.shannon()['Jensen-Shan'],
                        'Wave Hedges':lambda instance:instance.intersection()['Wave Hedges'],
                        'Jaccard':lambda instance:instance.inner()['Jaccard']} 
    
    scenes=rxr.open_rasterio(tif_fn,masked=True).squeeze()  
    ref_distance_dict={}
    for idx,row in tqdm(quadrats_gdf.iterrows(),total=quadrats_gdf.shape[0]):
        scene=scenes.rio.clip([mapping(row.geometry)],quadrats_gdf.crs)     
        sig_scene=list(map(signature_func_dict[signature],[scene.data]))[0]
        sig_scene=sig_scene/sig_scene.values.sum()
        
        ref_array=np.zeros(scene.data.shape)
        if idx==0:
            print(f'array shape:{ref_array.shape}')
        sig_ref=list(map(signature_func_dict[signature],[ref_array]))[0]
        sig_ref=sig_ref/sig_ref.values.sum()
        
        sig_ref,sig_scene=usda_df_process.complete_dataframe_rowcols([sig_ref,sig_scene])    
        distance_instance=usda_distance.Distances(sig_ref.to_numpy().flatten(),sig_scene.to_numpy().flatten()) 
        distance=list(map(distance_func_dict[distance_metirc],[distance_instance]))[0]
        ref_distance_dict[idx]=distance        

    quadrats_gdf_copy=quadrats_gdf.copy(deep=True)
    quadrats_gdf_copy['distance']=ref_distance_dict.values()   
    
    return quadrats_gdf_copy

class Stack():
    '''
    类似堆栈放入和取出的过程，用于数据临时存储。实现数据放入（push）、取出（pop）、是否为空（isEmpty）、已存入数据长度（size）和清空（clear）等操作
    '''
    def __init__(self):
        self.item = []
        self.obj=[]
    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []
        
class Pattern_segment_regionGrow():
    '''
    分类数据分割（segmentation），使用的是region growing algorithm，但是为signature标志特征和distance距离测量方式计算2维矩阵样方（scene）间的距离，根据指定阈值，判断是否合并
    '''
    def __init__(self,array,scene_x_num,scene_y_num,threshold,signature='class_clumpSize_histogram',distance_metirc='Jensen-Shan'):
        '''
        分类数据分割初始化值

        Parameters
        ----------
        array : 2darray
            2维数组，为分类数据.
        scene_x_num : int
            一个样方（scene）的宽度.
        scene_y_num : int
            一个样方（scene）的高度.
        threshold : float
            signature的距离阈值.
        signature : str, optional
            signature标记方法名称. The default is 'class_clumpSize_histogram'.
        distance_metirc : str, optional
            distance距离测量方法名称. The default is 'Jensen-Shan'.

        Returns
        -------
        None.

        '''
        self.h, self.w=array.shape
        self.scene_x_num=scene_x_num
        self.scene_y_num=scene_y_num
        
        self.quadrats,self.quadrat_w,self.quadrat_h=self.matrix2quadrat(array,scene_x_num,scene_y_num)
       # print(len(self.quadrats))
        self.quadrats_idx=np.array(range(self.quadrat_w*self.quadrat_h)).reshape(self.quadrat_h,self.quadrat_w)
        
        self.passedBy=np.zeros((self.quadrat_h,self.quadrat_w), np.double)
        self.currentRegion=0
        self.iterations=0  
        self.SEGS=np.zeros((self.quadrat_h,self.quadrat_w), dtype='int')
        self.stack=Stack()
        self.threshold=float(threshold)
        
        self.signature_func_dict={'class_clumpSize_histogram':lambda v: usda_signature.class_clumpSize_histogram(v,cc3d.connected_components(v,connectivity=8,return_N=False,out_dtype=np.uint64)),
                             'class_pairs_frequency':lambda v: usda_signature.class_co_occurrence(v),
                             'class_hierarchical_decomposition':lambda v: usda_signature.class_decomposition(v)}

        self.distance_func_dict={'Jensen-Shan':lambda instance:instance.shannon()['Jensen-Shan'],
                            'Wave Hedges':lambda instance:instance.intersection()['Wave Hedges'],
                            'Jaccard':lambda instance:instance.inner()['Jaccard']}  
        
        self.signature=signature
        self.distance_metirc=distance_metirc
        
    def matrix2quadrat(self,array,x_num,y_num):
        '''
        将2维矩阵划分为多个样方，用于计算signature

        Parameters
        ----------
        array : 2darray
            2维矩阵，为分类数据.
        x_num : int
            一个样方（scene）的宽度.
        y_num : int
            一个样方（scene）的高度.

        Returns
        -------
        quadrats : list[2darray]
            样方列表.
        quadrat_x_num : int
            样方的列数.
        quadrat_y_num : int
            样方的行数.

        '''
        h,w=array.shape
        quadrat_x_num=w//x_num
        quadrat_y_num=h//y_num  
        #print(quadrat_x_num,quadrat_y_num)
        array_clipped=array[:quadrat_y_num*y_num,:quadrat_x_num*x_num]
        quadrats= [M for Sub_array in np.split(array_clipped,quadrat_y_num, axis = 0) for M in np.split(Sub_array,quadrat_x_num, axis = 1)]
        
        return quadrats,quadrat_x_num,quadrat_y_num
        
    def limit(self, x,y):
        '''
        限制条件，邻接样方不能超过样方的行列数

        Parameters
        ----------
        x : int
            行索引.
        y : int
            列索引.

        Returns
        -------
        bool
            是否超出样方行列数.

        '''
        return  0<=x<self.quadrat_h and 0<=y<self.quadrat_w
    
    def getNeighbour(self, x0, y0):
        '''
        计算给定坐标点的邻接样方（坐标点）

        Parameters
        ----------
        x0 : int
            行索引.
        y0 : int
            列索引.

        Returns
        -------
        neighbour : list[tuple]
            邻接样方索引对列表.

        '''
        neighbour = []
        for i in (-1,0,1):
            for j in (-1,0,1):
                if (i,j) == (0,0): 
                    continue
                x = x0+i
                y = y0+j
                if self.limit(x,y):
                    neighbour.append((x,y))
        return neighbour
    
    def PassedAll(self):   
        '''
        是否超出样方数或这迭代次数

        Returns
        -------
        bool
            如果超出最大迭代次数，或者超出总样方数则停止计算.

        '''
        return self.iterations>200000 or np.count_nonzero(self.passedBy > 0) == self.quadrat_w*self.quadrat_h    
    
    def distance(self,x,y,x0,y0):
        '''
        由signature和distance的方法计算样方间的距离

        Parameters
        ----------
        x : int
            邻接样方行索引.
        y : int
            邻接样方列索引.
        x0 : int
            目标样方行索引.
        y0 : int
            目标样方列索引.

        Returns
        -------
        distance : float
            样方间的距离值.

        '''
        previous_idx=self.quadrats_idx[x0,y0]
        next_idx=self.quadrats_idx[x,y]
        
        previous_val=self.quadrats[previous_idx]
        next_val=self.quadrats[next_idx]
        
        sig_previous=list(map(self.signature_func_dict[self.signature],[previous_val]))[0]
        sig_next=list(map(self.signature_func_dict[self.signature],[next_val]))[0]
        
        sig_previous=sig_previous/sig_previous.values.sum()
        sig_next=sig_next/sig_next.values.sum()
        
        sig_previous,sig_next=usda_df_process.complete_dataframe_rowcols([sig_previous,sig_next])  
        distance_instance=usda_distance.Distances(sig_previous.to_numpy().flatten(),sig_next.to_numpy().flatten()) 
        distance=list(map(self.distance_func_dict[self.distance_metirc],[distance_instance]))[0]
        
        return distance
    
    def BFS(self, x0,y0):
        '''
        region growing algorithm

        Parameters
        ----------
        x0 : int
            目标样方行索引.
        y0 : int
            目标样方列索引.

        Returns
        -------
        None.

        '''
        regionNum=self.passedBy[x0,y0] 
        neighbours=self.getNeighbour(x0,y0)
        for x,y in neighbours:
            if self.passedBy[x,y] == 0 and self.distance(x,y,x0,y0)<self.threshold: 
                if(self.PassedAll()):
                    break;
                self.passedBy[x,y] = regionNum
                self.stack.push((x,y))
                self.prev_region_count+=1

    def seg_region_growing(self):
        '''
        region growing algorithm，主程序

        Returns
        -------
        None.

        '''
        
        for x0 in tqdm(range(self.quadrat_h)):
            for y0 in range(self.quadrat_w):   
                if self.passedBy[x0,y0] == 0 :
                    self.currentRegion += 1
                    self.passedBy[x0,y0] = self.currentRegion
                    self.stack.push((x0,y0))
                    self.prev_region_count=0
                    while not self.stack.isEmpty():                        
                        x,y = self.stack.pop()                        
                        self.BFS(x,y)
                        self.iterations+=1
                
        self.SEGS=self.passedBy.astype(int)         
        
    def quadrats_restore(self):
        '''
        将样方分割值返回为原数据形状大小。样方按分割后个区域值频数最大值作为最终值

        Returns
        -------
        None.

        '''
        seg_unique=np.unique(self.SEGS)
        seg_idx_mapping={i:self.quadrats_idx[self.SEGS==i] for i in seg_unique}
        seg_val={k:np.stack([self.quadrats[i] for i in v]).flatten() for k,v in seg_idx_mapping.items()}
        seg_most_frequent_val={k:np.bincount(v).argmax() for k,v in seg_val.items()}
        indexer=np.array([seg_most_frequent_val.get(i, -1) for i in range(self.SEGS.min(), self.SEGS.max() + 1)])
        quadrats_replaced=indexer[(self.SEGS - self.SEGS.min())]
        self.quadrats_restored=np.repeat(np.repeat(quadrats_replaced,self.scene_x_num,axis=1),self.scene_y_num,axis=0)

class Categorical_data_region_growing():
    '''
    分类数组分割，仅合并同一分类，相当于像素连通
    '''
    def __init__(self,array,th):
        self.im=array
        self.h, self.w=array.shape
        self.passedBy = np.zeros((self.h,self.w), np.double)
        self.currentRegion = 0
        self.iterations=0
        self.SEGS=np.zeros((self.h,self.w), dtype='int')
        self.stack = Stack()
        self.thresh=float(th)
        
    def limit(self, x,y):
        return  0<=x<self.h and 0<=y<self.w        
        
    def getNeighbour(self, x0, y0):
        neighbour = []
        for i in (-1,0,1):
            for j in (-1,0,1):
                if (i,j) == (0,0): 
                    continue
                x = x0+i
                y = y0+j
                if self.limit(x,y):
                    neighbour.append((x,y))
        return neighbour
    
    def distance(self,x,y,x0,y0):
        return abs(self.im[x,y]-self.im[x0,y0])
    
    def PassedAll(self):   
        return self.iterations>200000 or np.count_nonzero(self.passedBy > 0) == self.w*self.h    
    
    def BFS(self, x0,y0):
        regionNum = self.passedBy[x0,y0]   
        elems=[]
        elems.append(self.im[x0,y0])
        var=self.thresh
        neighbours=self.getNeighbour(x0,y0)
        for x,y in neighbours:
            if self.passedBy[x,y] == 0 and self.distance(x,y,x0,y0)<var: 
                if(self.PassedAll()):
                    break;
                self.passedBy[x,y] = regionNum
                self.stack.push((x,y))
                elems.append(self.im[x,y])
                self.prev_region_count+=1    
    
    def ApplyRegionGrow(self):
        randomseeds=[[self.h/2,self.w/2],
            [self.h/3,self.w/3],[2*self.h/3,self.w/3],[self.h/3-10,self.w/3],
            [self.h/3,2*self.w/3],[2*self.h/3,2*self.w/3],[self.h/3-10,2*self.w/3],
            [self.h/3,self.w-10],[2*self.h/3,self.w-10],[self.h/3-10,self.w-10]
                    ]
        
        np.random.shuffle(randomseeds)        
        for x0 in range (self.h):
            for y0 in range (self.w):   
                if self.passedBy[x0,y0] == 0 :
                    self.currentRegion += 1
                    self.passedBy[x0,y0] = self.currentRegion
                    self.stack.push((x0,y0))
                    self.prev_region_count=0
                    while not self.stack.isEmpty():
                        x,y = self.stack.pop()
                        self.BFS(x,y)
                        self.iterations+=1
                        
                    if(self.PassedAll()):
                        break
        self.SEGS=self.passedBy
