# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)

dependency: rasterio, earthpy, cv2, plotly (including for JupyterLab),rio_tiler
"""
from ._table_show import plotly_table
from ._table_show import print_html

from ._stats_charts import probability_graph
from ._graphic_drawing import demo_con_style
from ._graphic_drawing import demo_con_style_multiple

from ._colors import generate_colors
from ._colors import data_division
from ._colors import uniqueish_color
from ._colors import cmap_patch_build
from ._raster_show import bands_show
from ._image_process import image_exposure
from ._image_process import downsampling_blockFreqency
from ._image_process import img_rescale
from ._image_process import imgs_compression_cv
from ._raster_percentile_slider import percentile_slider
from ._imgs_show import img_struc_show
from ._imgs_show import imshow_label2darray
from ._imgs_show import plotly_scatterMapbox
from ._gif_show import animated_gif_show
from ._img_feature_extraction import Gaussion_blur
from ._img_feature_extraction import SIFT_detection
from ._img_feature_extraction import STAR_detection
from ._img_feature_extraction import feature_matching
from ._dynamic_streetView_visual_perception import DynamicStreetView_visualPerception
from ._knee_line_graph import knee_lineGraph
from ._moving_average_inflection import movingAverage_inflection
from ._moving_average_inflection import vanishing_position_length
from ._tile_show import Sentinel2_bandsComposite_show
from ._superpixel_segmentation_show import markBoundaries_layoutShow
from ._superpixel_segmentation_show import segMasks_layoutShow
from ._imgs_layout_show import imgs_layoutShow
from ._imgs_layout_show import imgs_layoutShow_FPList
from ._img_theme_color import img_theme_color
from ._img_theme_color import themeColor_impression
from ._chart_custom import boxplot_custom
from ._chart_custom import  histogram_3d

from ._gdf_plot import gdf_plot_annotate
from ._panorama_show import sphere_panorama_label

from ._plot_single_function import plot_single_function


__all__ = [
    "plotly_table",
    "print_html",
    "probability_graph",
    "demo_con_style",
    "demo_con_style_multiple",
    "generate_colors",
    "bands_show",
    "image_exposure",
    "downsampling_blockFreqency",
    "data_division",
    "percentile_slider",
    "uniqueish_color",
    "img_struc_show",
    "imshow_label2darray",
    "animated_gif_show",
    "Gaussion_blur",
    "STAR_detection",
    "plotly_scatterMapbox",
    "DynamicStreetView_visualPerception",
    "knee_lineGraph",
    "movingAverage_inflection",
    "vanishing_position_length",
    "Sentinel2_bandsComposite_show",
    "markBoundaries_layoutShow",
    "imgs_layoutShow",
    "imgs_layoutShow_FPList",
    "img_rescale",
    "img_theme_color",
    "themeColor_impression",
    "imgs_compression_cv",
    "boxplot_custom",
    "SIFT_detection",
    "feature_matching",
    "segMasks_layoutShow",
    "histogram_3d",
    "gdf_plot_annotate",
    "sphere_panorama_label",
    "plot_single_function",
    ]








