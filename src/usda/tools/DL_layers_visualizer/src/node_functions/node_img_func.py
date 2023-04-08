# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:37:41 2023

@author: richie bao
"""
import dearpygui.dearpygui as dpg


# Function of the "addition" node
def calc_imgarray(node_id,user_data=None):
    print('%%%%%%',user_data)
    # print('---###',node_id)
    result =dpg.get_value(node_id + "!Node_Img_Output_value")
    # Calculated value is set to the output socket
    # dpg.set_value((node_id + "!Node_Img_Output_value"), user_data)
    # print('---!!!',result)
    dpg.set_value((node_id + "!Node_Img_Output_value"),result)






