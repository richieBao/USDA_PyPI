# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:26:58 2022

@author: richie bao
"""
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
import os,pickle

class ERF_trainer:
    '''
    class - 用极端随机森林训练图像分类器
    '''
    def __init__(self,X,label_words,save_path):    
        print('Start training...')
        self.le=preprocessing.LabelEncoder()
        self.clf=ExtraTreesClassifier(n_estimators=100,max_depth=16,random_state=0) # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        y=self.encode_labels(label_words)
        self.clf.fit(np.asarray(X),y)
        with open(os.path.join(save_path,'ERF_clf.pkl'), 'wb') as f:  # 存储训练好的图像分类器模型
            pickle.dump(self.clf, f)   
        print("end training and saved estimator.")
            
    def  encode_labels(self,label_words):
        '''
        function - 对标签编码，及训练分类器
        '''
        self.le.fit(label_words)
        return np.array(self.le.transform(label_words),dtype=np.float64)
    
    def classify(self,X):
        '''
        function - 对未知数据的预测分类
        '''
        label_nums=self.clf.predict(np.asarray(X))
        label_words=self.le.inverse_transform([int(x) for x in label_nums])
        return label_words
    
    