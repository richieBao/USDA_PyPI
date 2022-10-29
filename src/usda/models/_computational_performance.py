# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:52:48 2022

@author: richie bao
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler    
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def PolynomialFeatures_regularization(X,y,regularization='linear'):
    '''
    function - 多项式回归degree次数选择，及正则化
    
    Params:
        X - 解释变量；array
        y - 响应变量；array
        regularization - 正则化方法， 为'linear'时，不进行正则化，正则化方法为'Ridge'和'LASSO'；string
        
    Returns:
        reg - model
    '''    

    
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=42)
    SS=StandardScaler()
    X_train_scaled=SS.fit_transform(X_train) 
    X_test_scaled=SS.fit_transform(X_test)

    degrees=np.arange(1,16,1)
    fig_row=3
    fig_col=degrees.shape[0]//fig_row
    fig, axs=plt.subplots(fig_row,fig_col,figsize=(21,12))
    r_squared_temp=0
    p=[(r,c) for r in range(fig_row) for c in range(fig_col)]
    i=0
    for d in degrees:
        if regularization=='linear':
            model=Pipeline([('poly', PolynomialFeatures(degree=d)),
                            ('regular', LinearRegression(fit_intercept=False))])
        elif regularization=='Ridge':
            model=Pipeline([('poly', PolynomialFeatures(degree=d)),
                            ('regular', Ridge())])            
        elif regularization=='LASSO':
            model=Pipeline([('poly', PolynomialFeatures(degree=d)),
                            ('regular', Lasso())])             
        
        reg=model.fit(X_train_scaled,y_train)
        x_=X_train_scaled.reshape(-1)
        print("训练数据集的-r_squared=%.6f,测试数据集的-r_squared=%.6f,对应的degree=%d"%(reg.score(X_train_scaled,y_train),reg.score(X_test_scaled,y_test) ,d))  
        print("系数:",reg['regular'].coef_)          
        print("_"*50)
        
        X_train_scaled_sort=np.sort(X_train_scaled,axis=0)

        axs[p[i][0]][p[i][1]].scatter(X_train_scaled.reshape(-1),y_train,c='black')
        axs[p[i][0]][p[i][1]].plot(X_train_scaled_sort.reshape(-1),reg.predict(X_train_scaled_sort),label='degree=%s'%d)
        axs[p[i][0]][p[i][1]].legend(loc='lower right', frameon=False)

        if r_squared_temp<reg.score(X_test_scaled,y_test): #knn-回归显著性检验（回归系数检验）
            r_squared_temp=reg.score(X_test_scaled,y_test) 
            d_temp=d   
        i+=1    

    plt.show()        
    model=Pipeline([('poly', PolynomialFeatures(degree=d_temp)),
                    ('linear', LinearRegression(fit_intercept=False))])
    reg=model.fit(X_train_scaled,y_train)    
    print("_"*50)
    print("在区间%s,最大的r_squared=%.6f,对应的degree=%d"%(degrees,reg.score(X_test_scaled,y_test) ,d_temp))  
    
    return reg

