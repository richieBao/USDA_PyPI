# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:59:31 2023

@author: richie bao

ref:DearPyGUI NodeEditor Template  https://codeberg.org/LuminousLizard/DearPyGUI_NodeEditor_Template
"""
import dearpygui.dearpygui as dpg
from src.node_editor import NodeEditor

dpg.create_context()
dpg.create_viewport(title="simple WGAN Analysis Visualizer",
                    width=1500,
                    height=768)

def callback_close_program(sender, data):
    exit(0)

def callback_show_debugger(sender, data):
    dpg.show_debug()
    
with dpg.font_registry():
    default_font = dpg.add_font("fonts/OpenSans-Regular.ttf", 15)
    
with dpg.viewport_menu_bar():
    dpg.add_menu_item(label="DEBUGGER", callback=callback_show_debugger)    
    # dpg.add_menu_item(label="Close", callback=callback_close_program)        
    dpg.bind_font(default_font)

nodeEditor=NodeEditor()

# Main Loop
dpg.show_viewport()
dpg.setup_dearpygui()
dpg.set_primary_window("NodeEditorWindow", True)
dpg.start_dearpygui()
dpg.destroy_context()

