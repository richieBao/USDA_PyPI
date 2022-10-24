# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 07:28:49 2022

@author: richie bao
"""
import pandas as pd
import numpy as np
import math
from scipy.stats import f  
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sympy import Matrix,pprint

def coefficient_of_determination(observed_vals,predicted_vals):   
    '''
    function - 回归方程的决定系数
    
    Params:
        observed_vals - 观测值（实测值）；list(float)
        predicted_vals - 预测值；list(float)
        
    Returns:
        R_square_a - 决定系数，由观测值和预测值计算获得；float
        R_square_b - 决定系数，由残差平方和和总平方和计算获得；float
    '''
    
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    # 观测值的离差平方和(总平方和，或总的离差平方和)
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    # 预测值的离差平方和
    pre_mean=vals_df.pre.mean()
    SS_reg=vals_df.pre.apply(lambda row:(row-pre_mean)**2).sum()
    # 观测值和预测值的离差积和
    SS_obs_pre=vals_df.apply(lambda row:(row.obs-obs_mean)*(row.pre-pre_mean), axis=1).sum()
    
    # 残差平方和
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
    
    # 判断系数
    R_square_a=(SS_obs_pre/math.sqrt(SS_tot*SS_reg))**2
    R_square_b=1-SS_res/SS_tot
            
    return R_square_a,R_square_b

def ANOVA(observed_vals,predicted_vals,df_reg,df_res):
    '''
    function - 简单线性回归方程-回归显著性检验（回归系数检验）
    
    Params:
        observed_vals - 观测值（实测值）；list(float)
        predicted_vals - 预测值；list(float)
        
    Returns:
        None
    '''
    
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    # 观测值的离差平方和(总平方和，或总的离差平方和)
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    
    # 残差平方和
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
   
    # 回归平方和
    SS_reg=vals_df.pre.apply(lambda row:(row-obs_mean)**2).sum()
    
    print("总平方和=%.6f,回归平方和=%.6f,残差平方和=%.6f"%(SS_tot,SS_reg,SS_res))
    print("总平方和=回归平方和+残差平方和：SS_tot=SS_reg+SS_res=%.6f+%.6f=%.6f"%(SS_reg,SS_res,SS_reg+SS_res))
    
    Fz=(SS_reg/df_reg)/(SS_res/df_res)
    print("F-分布统计量=%.6f;p-value=%.6f"%(Fz,f.sf(Fz,df_reg,df_res)))
    
def confidenceInterval_estimator_LR(x,sample_num,X,y,model,confidence=0.05):
    '''
    function - 简单线性回归置信区间估计，及预测区间
    
    Params:
        x - 自变量取值；float
        sample_num - 样本数量；int
        X - 样本数据集-自变量；list(float)
        y - 样本数据集-因变量；list(float)
        model -使用sklearn获取的线性回归模型；model
        confidence -  置信度。The default is 0.05" ；float
    
    Returns:
    
       CI - 置信区间；list(float)
    '''
    
    X_=X.reshape(-1)
    X_mu=X_.mean()
    s_xx=(X_-X_mu)**2
    S_xx=s_xx.sum()
    ss_res=(y-model.predict(X))**2
    SS_res=ss_res.sum()
    probability_val=f.ppf(q=1-confidence,dfn=1, dfd=sample_num-2) # dfn=1, dfd=sample_num-2
    CI=[math.sqrt(probability_val*(1/sample_num+(x-X_mu)**2/S_xx)*SS_res/(sample_num-2)) for x in X_]
    y_pre=model.predict(X)
    
    fig, ax=plt.subplots(figsize=(10,10))
    ax.plot(X_,y,'o',label='observations/ground truth',color='r')
    ax.plot(X_,y_pre,'o-',label='linear regression prediction')
    ax.plot(X_,y_pre-CI,'--',label='y_lower')
    ax.plot(X_,y_pre+CI,'--',label='y_upper')
    ax.fill_between(X_, y_pre-CI, y_pre+CI, alpha=0.2,label='95% confidence interval')    
      
    # 给定值的预测区间
    x_ci=math.sqrt(probability_val*(1/sample_num+(x-X_mu)**2/S_xx)*SS_res/(sample_num-2))
    x_pre=model.predict(np.array([x]).reshape(-1,1))[0]
    x_lower=x_pre-x_ci
    x_upper=x_pre+x_ci
    print("x prediction=%.6f;confidence interval=[%.6f,%.6f]"%(x_pre,x_lower,x_upper))
    ax.plot(x,x_pre,'x',label='x_prediction',color='r',markersize=20)
    ax.arrow(x, x_pre, 0, x_upper-x_pre, head_width=0.3, head_length=2,color="gray",linestyle="--" ,length_includes_head=True)
    ax.arrow(x, x_pre, 0, x_lower-x_pre, head_width=0.3, head_length=2,color="gray",linestyle="--" ,length_includes_head=True)
        
    ax.set(xlabel='temperature',ylabel='ice tea sales')
    ax.legend(loc='upper left', frameon=False)    
    plt.show()                  
    return CI    

def correlationAnalysis_multivarialbe(df):
    '''
    function - DataFrame数据格式，成组计算pearsonr相关系数
    
    Params:
        df - DataFrame格式数据集；DataFrame(float)
    
    Returns:
        p_values - P值；DataFrame(float)
        correlation - 相关系数；DataFame(float)
    '''
    
    df=df.dropna()._get_numeric_data()
    df_cols=pd.DataFrame(columns=df.columns)
    p_values=df_cols.transpose().join(df_cols, how='outer')
    correlation=df_cols.transpose().join(df_cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            p_values[r][c]=round(pearsonr(df[r], df[c])[1], 4)
            correlation[r][c]=round(pearsonr(df[r], df[c])[0], 4)
            
    return p_values,correlation
    
def coefficient_of_determination_correction(observed_vals,predicted_vals,independent_variable_n):
    '''
    function - 回归方程修正自由度的判定系数
    
    Params:
        observed_vals - 观测值（实测值）；list(float)
        predicted_vals - 预测值；list(float)
        independent_variable_n - 自变量个数；int
        
    Returns:
        R_square_correction - 正自由度的判定系数；float
    '''
    
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    # 观测值的离差平方和(总平方和，或总的离差平方和)
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    
    # 残差平方和
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
    
    # 判断系数
    sample_n=len(observed_vals)
    R_square_correction=1-(SS_res/(sample_n-independent_variable_n-1))/(SS_tot/(sample_n-1))
            
    return R_square_correction   
    
def ANOVA_multivarialbe(observed_vals,predicted_vals,independent_variable_n,a_i,X):
    '''
    function - 多元线性回归方程-回归显著性检验（回归系数检验），全部回归系数的总体检验，及单个回归系数的检验
    
    Paras:    
        observed_vals - 观测值（实测值）；list(float)
        predicted_vals - 预测值；list(list)
        independent_variable_n - 自变量个数；int
        a_i - 偏相关系数列表；list(float)
        X - 样本数据集_自变量；array(numpy)
        
    Returns:
        None
    '''
    
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    # 总平方和，或总的离差平方和
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    
    # 残差平方和
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
   
    # 回归平方和
    SS_reg=vals_df.pre.apply(lambda row:(row-obs_mean)**2).sum()
    
    # 样本个数
    n_s=len(observed_vals)
    dfn=independent_variable_n
    dfd=n_s-independent_variable_n-1
    
    # 计算全部回归系数的总体检验统计量
    F_total=((SS_tot-SS_res)/dfn)/(SS_res/dfd)
    print("F-分布统计量_total=%.6f;p-value=%.6f"%(F_total,f.sf(F_total,dfn,dfd)))
    
    # 逐个计算单个回归系数的检验统计量
    X=np.insert(X,0,1,1)
    X_m=Matrix(X)
    M_inverse=(X_m.T*X_m)**-1
    C_jj=M_inverse.row(1).col(1)[0]
    pprint(C_jj)
    
    F_ai_list=[]
    i=0
    for a in a_i:
        F_ai=(a**2/C_jj)/(SS_res/dfd)
        F_ai_list.append(F_ai)
        print("a%d=%.6f时，F-分布统计量_=%.6f;p-value=%.6f"%(i,a,F_ai,f.sf(F_total,1,dfd)))
        i+=1    
    
    
def confidenceInterval_estimator_LR_multivariable(X,y,model,confidence=0.05):
    '''
    function - 多元线性回归置信区间估计，及预测区间
    
    Params:
        X - 样本自变量；DataFrame数据格式
        y - 样本因变量；list(float)
        model - 多元回归模型；model
        confidence - 置信度，The default is 0.05；float
    
    return:
        CI- 预测值的置信区间；list(float)
    '''
    
    # 根据指定数目，划分列表的函数
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    X_deepCopy=X.copy(deep=True) # 不进行深度拷贝，如果传入的参数变量X发生了改变，则该函数外部的变量值也会发生改变
    columns=X_deepCopy.columns
    n_v=len(columns)
    n_s=len(y)
    
    # 求S，用于马氏距离的计算
    SD=[]
    SD_name=[]
    for col_i in columns:
        i=0
        for col_j in columns:
            SD_column_name=col_i+'S'+str(i)
            SD_name.append(SD_column_name)
            if col_i==col_j:
                X_deepCopy[SD_column_name]=X_deepCopy.apply(lambda row: (row[col_i]-X_deepCopy[col_j].mean())**2,axis=1)
                SD.append(X_deepCopy[SD_column_name].sum())
            else:
                X_deepCopy[SD_column_name]=X_deepCopy.apply(lambda row: (row[col_i]-X_deepCopy[col_i].mean())*(row[col_j]-X_deepCopy[col_j].mean()),axis=1)
                SD.append(X_deepCopy[SD_column_name].sum())                
            i+=1
    M=Matrix(list(chunks(SD,n_v)))
    
    # 求S的逆矩阵
    M_invert=M**-1
    # pprint(M_invert)
    M_invert_list=[M_invert.row(row).col(col)[0] for row in range(n_v) for col in range(n_v)]
    X_mu=[X_deepCopy[col].mean() for col in columns]
    
    # 求马氏距离的平方
    SD_array=X_deepCopy[SD_name].to_numpy()    
    D_square_list=[sum([x*y for x,y in zip(SD_selection,M_invert_list)])*(n_s-1) for SD_selection in SD_array]    
    
    # 计算CI-预测值的置信区间
    print(columns)
    ss_res=(y-model.predict(X_deepCopy[columns].to_numpy()))**2
    SS_res=ss_res.sum()
    print(SS_res)
    probability_val=f.ppf(q=1-confidence,dfn=1, dfd=n_s-n_v-1) 
    CI=[math.sqrt(probability_val*(1/n_s+D_square/(n_s-1))*SS_res/(n_s-n_v-1)) for D_square in D_square_list]

    return CI   
    

    