# Copyright 2021 LuminousLizard
# Licensed under the MIT-License

import dearpygui.dearpygui as dpg
from src.chain_update import func_chain_update, func_link_destroyed
from src.nodes import *

LastNodePosition = [100, 100]


# Destroy window if closed
def callback_close_window(sender):
    dpg.delete_item(sender)

class NodeEditor:
    def __init__(self):
        with dpg.window(tag="NodeEditorWindow",
                        label="Node editor",
                        width=1000,
                        height=700,
                        pos=[50, 50],
                        menubar=True,
                        on_close=callback_close_window):
            # Add a menu bar to the window
            with dpg.menu_bar(label="MenuBar"):
                with dpg.menu(label="Input/Output nodes"):
                    dpg.add_menu_item(tag="Menu_AddNode_InputFloat",
                                      label="Input float",
                                      callback=callback_add_node,
                                      user_data="Input_Float")
                    dpg.add_menu_item(tag="Menu_AddNode_OutputFloat",
                                      label="Output_Float",
                                      callback=callback_add_node,
                                      user_data="Output_Float")

                with dpg.menu(label="Math nodes"):
                    dpg.add_menu_item(tag="Menu_AddNode_Addition",
                                      label="Addition",
                                      callback=callback_add_node,
                                      user_data="Addition")
                    dpg.add_menu_item(tag="Menu_AddNode_Subtraction",
                                      label="Subtraction",
                                      callback=callback_add_node,
                                      user_data="Subtraction")
                    dpg.add_menu_item(tag="Menu_AddNode_Multiplication",
                                      label="Multiplication",
                                      callback=callback_add_node,
                                      user_data="Multiplication")
                    dpg.add_menu_item(tag="Menu_AddNode_Division",
                                      label="Division",
                                      callback=callback_add_node,
                                      user_data="Division")  
                    
            # with dpg.group(horizontal=True):
            #     dpg.add_text("Status:")
            #     dpg.add_text(tag="InfoBar")    
                 
            #------------------------------------------------------------------   
            button_width=188
            with dpg.group(horizontal=True, width=0):        
                with dpg.child_window(width=200,):     
                    button_0_tag=button_selectable('red',2,'button_0_theme')
                    
                    dpg.add_button(tag="b_0",label='drop selected nodes',callback=drop_selected_nodes,user_data='NodeEditor',width=button_width)
                    dpg.add_button(tag='b_1',label="Input Float", callback=callback_add_node,user_data="Input_Float",width=button_width)
                    dpg.add_button(tag='b_2',label="Output Float", callback=callback_add_node,user_data="Output_Float",width=button_width)
                    dpg.add_button(tag='b_3',label="Addition", callback=callback_add_node,user_data="Addition",width=button_width)
                    dpg.add_button(tag='b_4',label="Subtraction", callback=callback_add_node,user_data="Subtraction",width=button_width)
                    dpg.add_button(tag='b_5',label="Multiplication", callback=callback_add_node,user_data="Multiplication",width=button_width)
                    dpg.add_button(tag='b_6',label="Division", callback=callback_add_node,user_data="Division",width=button_width)
                    
                    button_theme_0_lst=['b_1','b_2','b_3','b_4','b_5','b_6']    
                    for button in button_theme_0_lst:
                        dpg.bind_item_theme(button, button_0_tag)
                        
                    button_1_tag=button_selectable('blue',2,'button_1_theme')
                    dpg.bind_item_theme('b_0', button_1_tag)
                    
                    button_2_tag=button_selectable('orange',2,'button_2_theme')                    
                    dpg.add_button(tag='b_7',label="Input Image", callback=callback_add_node,user_data="Image",width=button_width)
                    dpg.add_button(tag='b_8',label="Load WGAN Model_64", callback=callback_add_node,user_data="WGAN64_Model",width=button_width)
                    
                    button_theme_2_lst=['b_7','b_8'] 
                    for button in button_theme_2_lst:
                        dpg.bind_item_theme(button, button_2_tag)
                        
                    button_3_tag=button_selectable('green',2,'button_3_theme') 
                    dpg.add_button(tag='b_9',label="File Path Selector", callback=callback_add_node,user_data="File_Path",width=button_width)    
                    button_theme_3_lst=['b_9']
                    for button in button_theme_3_lst:
                        dpg.bind_item_theme(button, button_3_tag)                    
                    
                    
                    
                    
                #------------------------------------------------------------------                         
                # Add node editor to the window
                with dpg.child_window(autosize_x=True,):
                    with dpg.node_editor(tag="NodeEditor",
                                         # Function call for updating all nodes if a new link is created
                                         callback=func_chain_update,
                                         # Function call for updating if a link is destroyed
                                         delink_callback=func_link_destroyed,
                                         minimap=True, 
                                         minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                        pass

            with dpg.handler_registry():
                dpg.add_mouse_click_handler(callback=save_last_node_position)                
                
            # End note editor
            #------------------------------------------------------------------

# Saving the position of the last selected node
def save_last_node_position():
    global LastNodePosition
    if dpg.get_selected_nodes("NodeEditor") == []:
        pass
    else:
        LastNodePosition = dpg.get_item_pos(dpg.get_selected_nodes("NodeEditor")[0])


def callback_add_node(sender, app_data, user_data):
    # print(app_data)
    function_dict = {
        "Input_Float": node_input_float.add_node_input_float,
        "Output_Float": node_output_float.add_node_output_float,
        "Addition": node_addition.AddNodeAddition,
        "Subtraction": node_subtraction.add_node_subtraction,
        "Multiplication": node_multiplication.add_node_multiplication,
        "Division": node_division.add_node_division,
        "Image":node_img.add_node_image,
        "WGAN64_Model":node_WGAN64_model.add_node_wgan_64,
        "File_Path":node_file_path.add_node_file_path,
    }
    function_dict[user_data](LastNodePosition)

def _hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (255*v, 255*t, 255*p)
    if i == 1: return (255*q, 255*v, 255*p)
    if i == 2: return (255*p, 255*v, 255*t)
    if i == 3: return (255*p, 255*q, 255*v)
    if i == 4: return (255*t, 255*p, 255*v)
    if i == 5: return (255*v, 255*p, 255*q)
    
def button_selectable(button_color,button_size,tag):
    color_dict={'red':0,'orange':1,'green':2,'aqua':3,'blue':4,'violet':5,'pink':6}
    button_color=color_dict[button_color]
    
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(button_color/7.0, 0.6, 0.6))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(button_color/7.0, 0.8, 0.8))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(button_color/7.0, 0.7, 0.7))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, button_size*5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, button_size*3, button_size*3)  
            
    return tag
           
def drop_selected_nodes(sender, app_data, user_data):
    selected_nodes=dpg.get_selected_nodes(user_data)
    for node in selected_nodes:
        dpg.delete_item(node)