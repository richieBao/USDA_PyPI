# Copyright 2021 LuminousLizard
# Licensed under the MIT-License

from src.node_functions import *


def linking_node_function(node_tag,user_data=None):
    link_dict = {
        "Node_Addition": node_addition_func.calc_addition,
        "Node_Subtraction": node_subtraction_func.calc_subtraction,
        "Node_Multiplication": node_multiplication_func.calc_multiplication,
        "Node_Division": node_division_func.calc_division,
        "Node_Img": node_img_func.calc_imgarray,
    }
    for key in link_dict:
        if key in node_tag[1]:
            if user_data is not None:
                link_dict[key](node_tag[0],user_data)
            else:
                link_dict[key](node_tag[0])
