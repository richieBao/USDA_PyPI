# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 08:31:59 2022

@author: richie bao
"""
import datetime

def start_time():    
    '''
    function-计算当前时间
    '''
    start_time=datetime.datetime.now()
    print("start time:",start_time)
    return start_time

def duration(start_time):    
    '''
    function-计算持续时间
    
    Params:
    start_time - 开始时间；datatime
    '''
    end_time=datetime.datetime.now()
    print("end time:",end_time)
    duration=(end_time-start_time).seconds/60
    print("Total time spend:%.2f minutes"%duration)

