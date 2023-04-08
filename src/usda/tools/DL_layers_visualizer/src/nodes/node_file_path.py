# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:42:28 2023

@author: richie bao
"""
import dearpygui.dearpygui as dpg
from random import randint
from src.chain_update import func_chain_update
from src.util import *

# Add simple input node for an int value with hard coded ID "1"
def add_node_file_path(user_data):
    # Create random ID and check that the ID does not exist yet for this node type        
    node_type="!Node_Filepath"    
    random_id=unique_tag_randint(node_type)    
    
    with dpg.file_dialog(directory_selector=False, show=False,callback=filedialog_callback, tag=random_id+"!file_dialog_id", width=700 ,height=400,user_data=[random_id,node_type]): 
        dpg.add_file_extension("Source files (*.pth *.pkl *.ckp){.pth,.pkl,.ckp}",color=(255, 0, 255, 255))        
    dpg.add_file_dialog(directory_selector=True, show=False,callback=filedialog_callback, tag=random_id+"!directory_dialog_id", width=700 ,height=400,user_data=[random_id,node_type])

    with dpg.node(tag=str(random_id) + node_type,
                  parent="NodeEditor",
                  label="File Path Selector",
                  pos=user_data):
        with dpg.node_attribute(tag=str(random_id) + "!Node_FilePath_Output",): # attribute_type=dpg.mvNode_Attr_Output
            dpg.add_radio_button(("file selector", "directory selector"),tag=random_id+'!Node_FilePath_radio',horizontal=True,callback=radio_callback,user_data=[random_id,node_type])
            dpg.add_button(tag=random_id+'!b_file_selector',label="File Selector", callback=lambda: dpg.show_item(random_id+"!file_dialog_id"))
            
def filedialog_callback(sender, app_data, user_data):  
    random_id,node_type=user_data
    tag_lst=[str(random_id) + node_type+ "_Output",random_id+node_type+'_Output_value']   
    delete_listed_items(tag_lst)
    
    parent_tag=str(random_id)+ "!Node_FilePath_Output"
    if sender.split('!')[1]=='file_dialog_id':
        path=list(app_data['selections'].values())[0]      
    elif sender.split('!')[1]=='directory_dialog_id':        
        path=app_data['file_path_name']  
     
    with dpg.node_attribute(tag=str(random_id) + node_type+ "_Output", attribute_type=dpg.mvNode_Attr_Output,parent=str(random_id) +node_type):
        dpg.add_text(path,tag=random_id+node_type+'_Output_value',label='path',bullet=True)    
 
def radio_callback(sender,app_data,user_data):
    random_id,node_type=user_data
    tag_lst=[random_id+'!b_file_selector',random_id+'!b_directory_selector',str(random_id) + node_type+ "_Output",random_id+node_type+'_Output_value']   
    delete_listed_items(tag_lst)

    parent_tag=random_id+"!Node_FilePath_Output"
    if app_data=="file selector":
        dpg.add_button(tag=random_id+'!b_file_selector',label="File Selector", callback=lambda: dpg.show_item(random_id+"!file_dialog_id"),parent=parent_tag)
    elif  app_data=="directory selector":
        dpg.add_button(tag=random_id+'!b_directory_selector',label="Directory Selector", callback=lambda: dpg.show_item(random_id+"!directory_dialog_id"),parent=parent_tag)
            
