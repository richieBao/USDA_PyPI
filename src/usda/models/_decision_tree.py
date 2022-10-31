# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:58:21 2022

@author: richie bao
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score 

def decisionTree_structure(X,y,criterion='entropy',cv=None,figsize=(6, 6)):
    '''
    function - 使用决策树分类，并打印决策树流程图表。迁移于Sklearn的'Understanding the decision tree structure', https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    
    Params:
        X - 数据集-特征值（解释变量）；ndarray
        y- 数据集-类标/标签(响应变量)；ndarray
        criterion - DecisionTreeClassifier 参数，衡量拆分的质量，即衡量哪一项检测最能减少分类的不确定性；string
        cv - cross_val_score参数，确定交叉验证分割策略，默认值为None，即5-fole(折)的交叉验证；int
        
    Returns:
        clf - 返回决策树模型
    '''   
    
    #X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=0)
    X_train,y_train=X,y
    clf=DecisionTreeClassifier(criterion=criterion,max_leaf_nodes=3, random_state=0)
    clf.fit(X_train, y_train)        
    
    n_nodes=clf.tree_.node_count
    children_left=clf.tree_.children_left
    children_right=clf.tree_.children_right
    feature=clf.tree_.feature
    threshold=clf.tree_.threshold    
    print("n_nodes:{n_nodes},\nchildren_left:{children_left},\nchildren_right={children_right},\nthreshold={threshold}".format(n_nodes=n_nodes,children_left=children_left,children_right=children_right,threshold=threshold))
    print("_"*50)
    
    node_depth=np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves=np.zeros(shape=n_nodes, dtype=bool)
    stack=[(0, 0)]  # start with the root node id (0) and its depth (0)    

    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has {n} nodes and has "
          "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: "
                  "go to node {left} if X[:, {feature}] <= {threshold} "
                  "else to node {right}.".format(
                      space=node_depth[i] * "\t",
                      node=i,
                      left=children_left[i],
                      feature=feature[i],
                      threshold=threshold[i],
                      right=children_right[i]))   
            
    plt.figure(figsize=figsize)      
    tree.plot_tree(clf)
    plt.show()
    
    CV_scores=cross_val_score(clf,X,y, cv=cv)
    print('cross_val_score:\n',CV_scores) # 交叉验证每次运行的估计器得分数组   
    print("%0.2f accuracy with a standard deviation of %0.2f" % (CV_scores.mean(), CV_scores.std())) # 同时给出了平均得分，和标准差
    return clf