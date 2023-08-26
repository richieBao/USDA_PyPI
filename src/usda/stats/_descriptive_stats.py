# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:46:10 2022

@author: richie bao
"""
import pandas as pd
import numpy as np
import seaborn as sns 

from scipy import stats

def frequency_bins(df,bins,field):    
    '''
    function - 频数分布计算
    
    Params:
        df - 单列（数值类型）的DataFrame数据；DataFrame(Pandas)
        bins - 配置分割区间（组距）；range()，例如：range(0,600+50,50)
        field - 字段名；string
        
    Returns:
        df_fre - 统计结果字段包含：index（为bins）、fre、relFre、median和fre_percent%；DataFrame
    '''   
    # A-组织数据
    column_name=df.columns[0]
    column_bins_name=df.columns[0]+'_bins'
    df[column_bins_name]=pd.cut(x=df[column_name],bins=bins,right=False)  # 参数right=False指定为包含左边值，不包括右边值。
    df_bins=df.sort_values(by=[column_name])  # 按照分割区间排序
    df_bins.set_index([column_bins_name,df_bins.index],drop=False,inplace=True)  # 以price_bins和原索引值设置多重索引，同时配置drop=False参数保留原列。
    
    # B-频数计算
    dfBins_frequency=df_bins[column_bins_name].value_counts()  # dropna=False  
    dfBins_relativeFrequency=df_bins[column_bins_name].value_counts(normalize=True)  # 参数normalize=True将计算相对频数(次数) dividing all values by the sum of values
    dfBins_freqANDrelFreq=pd.DataFrame({'fre':dfBins_frequency,'relFre':dfBins_relativeFrequency})
    
    # C-组中值计算
    df_bins[field]=df_bins[field].astype(float)
    dfBins_median=df_bins.groupby(level=0).median(numeric_only=True)
    dfBins_median.rename(columns={column_name:'median'},inplace=True)
    
    # D-合并分割区间、频数计算和组中值的DataFrame格式数据。
    df_fre=dfBins_freqANDrelFreq.join(dfBins_median).sort_index().reset_index()  # 在合并时会自动匹配index
    
    # E-计算频数比例
    df_fre['fre_percent%']=df_fre.apply(lambda row:row['fre']/df_fre.fre.sum()*100,axis=1)
    
    return df_fre

def comparisonOFdistribution(df,field,bins=100):
    '''
    funciton-数据集z-score概率密度函数分布曲线（即观察值/实验值 observed/empirical data）与标准正态分布(即理论值 theoretical set)比较
    
    Params:
        df - 包含待分析数据集的DataFrame格式类型数据；DataFrame(Pandas)
        field - 指定分析数据数据（DataFrame格式）的列名；string
        bins - 指定频数宽度，为单一整数代表频数宽度（bin）的数量；或者列表，代表多个频数宽度的列表。The default is 100；int;list(int)
        
    Returns:
        None
    '''  
    df_field_mean=df[field].mean()
    df_field_std=df[field].std()
    print("mean:%.2f, SD:%.2f"%(df_field_mean,df_field_std))

    df['field_norm']=df[field].apply(lambda row: (row-df_field_mean)/df_field_std)  # 标准化价格(标准计分，z-score)，或者使用`from scipy.stats import zscore`方法

    # 验证z-score，标准化后的均值必为0， 标准差必为1.0
    df_fieldNorm_mean=df['field_norm'].mean()
    df_fieldNorm_std=df['field_norm'].std()
    print("norm_mean:%.2f, norm_SD:%.2f"%(df_fieldNorm_mean,df_fieldNorm_std))
  
    sns.histplot(df['field_norm'], bins=bins,kde=True,stat="density",linewidth=0,color='r')

    s=np.random.normal(0, 1, len(df[field]))
    sns.histplot(s, bins=bins,kde=True,stat="density",linewidth=0,color='b')
    
def xdas_stats(xdas,exclulde=None):
    stats4pop={}
    exception_k=[]
    for k,downsampled in xdas.items(): 
        grain=[round(i,3) for i in downsampled.rio.resolution()]
        if exclulde is None:
            data=downsampled.data
            data_flatten=data.reshape(-1)
        else:
            data=np.setdiff1d(downsampled.data,exclulde)
            data_flatten=data.reshape(-1)

        try:
            stats4pop[grain[0]]=dict(
                # shape=data.shape,
                n=len(data_flatten),
                max=np.max(data_flatten),
                min=np.min(data_flatten),
                sampleRange=np.ptp(data_flatten),
                median=np.median(data_flatten),
                mean=np.mean(data_flatten),
                harmonic_mean=stats.hmean(data_flatten),
                std=np.std(data_flatten),
                var=np.var(data_flatten),
                nanstd=np.nanstd(data_flatten),
                nanvar=np.nanvar(data_flatten),
                skew=stats.skew(data_flatten),
                kurtosis=stats.kurtosis(data_flatten),
                entropy=stats.entropy(data_flatten),            
                )
        except:
            exception_k.append(k)

        stats4pop_df=pd.DataFrame.from_dict(stats4pop)

    return stats4pop_df,exception_k    