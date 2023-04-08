# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:36:59 2023

@author: richie bao
"""
import dearpygui.dearpygui as dpg
import random
import numpy as np
from src.chain_update import func_chain_update
from src.util import *
from scipy.stats import wasserstein_distance

from fastai.vision.all import *

def add_node_WGAD_PDF_plot(user_data):
    # Create random ID and check that the ID does not exist yet for this node type  
    node_type = "!Node_WGAN64_D_PDF_plot"    
    random_id=unique_tag_randint(node_type)
        
    with dpg.node(tag=random_id + node_type,
                  parent="NodeEditor",
                  label="WGAN64_D_PDF_plot",
                  pos=user_data):               
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Input1"):
            dpg.add_input_text(tag=str(random_id) + node_type + "_Input1_value",
                                label="WGAN Model ID",
                                width=100,
                                enabled=False,
                                readonly=True,
                                )
            
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Input2"):
            dpg.add_input_text(tag=str(random_id) + node_type + "_Input2_value",
                                label="Images_batch_ID",
                                width=100,
                                enabled=False,
                                readonly=True,
                                )             
            
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Input3"):   
            dpg.add_drag_int(tag=str(random_id) + node_type + "_Input3_value",
                             label="drag int 0-1000", 
                             # format="%d%%",
                             min_value=10,
                             max_value=1000,
                             default_value=100,
                             width=100,)
            
        with dpg.node_attribute(tag=str(random_id) + node_type + "_Cal", ):     
            dpg.add_button(label='cal PDF of D',
                           tag=random_id+'!cal_Dpdf',
                           callback=button_cal_pdf_D,
                           user_data=[random_id,node_type],
                           arrow=True,
                           direction=1,)        
            
def button_cal_pdf_D(sender,app_data,user_data):
    random_id,node_type=user_data
    
    nz=dpg.get_value(str(random_id) + node_type + "_Input3_value")
    print(nz)     
    
    model_ID=dpg.get_value(str(random_id) + node_type + "_Input1_value")
    learn=gvals[model_ID]
    
    imgs_batch_ID=dpg.get_value(str(random_id) + node_type + "_Input2_value")
    imgs_batch=gvals[imgs_batch_ID]  
    
    netD=learn.model.critic
    netD=netD.to(torch.device('cuda'))
    results=netD(imgs_batch).detach().cpu()

    counts, bin_edges=np.histogram(results, bins=10,density=True)
    pdf=counts/sum(counts)    
    x=bin_edges[1:]
    
    ngpu=1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    netG=learn.model.generator
    netG=netG.to(torch.device('cuda'))    

    fixed_noise=torch.randn(imgs_batch.shape[0], nz,device=device)    
    fake=netG(fixed_noise)
    fake_results=netD(fake).detach().cpu()
    fake_counts, fake_bin_edges = np.histogram(fake_results, bins=10,density=True)
    fake_pdf=fake_counts/sum(fake_counts)    
    fake_x=fake_bin_edges[1:]
        
    gvals['netD_result'+random_id]={'real_D':results,'fale_D':fake_results}
    
    EMDistance=wasserstein_distance(x,fake_x,pdf,fake_pdf)
    
    # create plot
    with dpg.plot(label="pdf_Line Series", height=400,parent=str(random_id) + node_type + "_Cal"):
        # optionally create legend
        dpg.add_plot_legend()
        # REQUIRED: create x and y axes
        dpg.add_plot_axis(dpg.mvXAxis, label="x")        
        with dpg.plot_axis(dpg.mvYAxis, label="y"):
            # series belong to a y axis
            dpg.add_line_series(x, counts, label="real netD-actual PDF")    
            dpg.add_line_series(x, pdf, label="real netD-normalised_PDF")   
            
            dpg.add_line_series(fake_x,fake_counts, label="fake netD-actual PDF")    
            dpg.add_line_series(fake_x,fake_pdf, label="fake netD-normalised_PDF")     
            
    dpg.add_text(f'wasserstein-1 distance={EMDistance}', color=(255, 0, 255),parent=str(random_id) + node_type + "_Cal")
       
    with dpg.node_attribute(tag=str(random_id) + node_type+ "_Output", attribute_type=dpg.mvNode_Attr_Output,parent=str(random_id) + node_type):
        dpg.add_text(tag=str(random_id) + node_type+ "_Output_value",
                      label="netD real and fake results",
                      default_value='netD_result'+random_id,
                      bullet=True)
    
    


        
