# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:17:29 2023

@author: richie bao
"""
import dearpygui.dearpygui as dpg
from random import randint
from src.chain_update import func_chain_update
from src.util import *

# Add simple input node for an int value with hard coded ID "1"
def add_node_input_text(user_data):
    # Create random ID and check that the ID does not exist yet for this node type
    node_type="!Node_InputText"  
    random_id=unique_tag_randint(node_type)   

    with dpg.node(tag=str(random_id) + node_type,
                  parent="NodeEditor",
                  label="Input text",
                  pos=user_data):
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Input1"):
            dpg.add_input_text(tag=str(random_id) + node_type + "_Input1_value",
                                label="Text Input",
                                width=200,
                                enabled=True,
                                readonly=False,
                                ) 
            
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Cal", ):     
            dpg.add_button(label='Text Output',
                           tag=random_id+'!Text_Output',
                           callback=button_test_output,
                           user_data=[random_id,node_type],
                           arrow=True,
                           direction=1,)        
            
def button_test_output(sender,app_data,user_data):
    random_id,node_type=user_data
    
    text_input=dpg.get_value(str(random_id) + node_type + "_Input1_value")
    with dpg.node_attribute(tag=str(random_id) + node_type+ "_Output", attribute_type=dpg.mvNode_Attr_Output,parent=str(random_id) + node_type):
        dpg.add_text(tag=str(random_id) + node_type+ "_Output_value",
                     label="Text output",
                     default_value=text_input,
                     bullet=True)                             
