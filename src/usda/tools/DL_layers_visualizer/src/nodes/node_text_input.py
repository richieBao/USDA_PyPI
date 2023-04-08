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
    node_type="!Node_Filepath"    
    random_id=unique_tag_randint(node_type)   

    with dpg.node(tag=str(random_id) + "!Node_InputText",
                  parent="NodeEditor",
                  label="Input text",
                  pos=user_data):
        with dpg.node_attribute(tag=str(random_id) + "!Node_Text"): 
            
            
        
        
        # with dpg.node_attribute(tag=str(random_id) + "!Node_InputFloat_Output", attribute_type=dpg.mvNode_Attr_Output):
        #     dpg.add_input_float(tag=str(random_id) + "!Node_InputFloat_Output_value",
        #                         label="Float value",
        #                         width=150,
        #                         default_value=0,
        #                         callback=func_chain_update)

