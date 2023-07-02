# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from ._base import load_sales_data_cartoon_database
from ._base import load_ramen_price_cartoon_statistic
from ._base import load_bowling_contest_cartoon_statistic
from ._base import load_test_score_cartoon_statistic
from ._base import baiduPOI_dataCrawler
from ._base import baiduPOI_dataCrawler_circle
from ._base import baiduPOI_batchCrawler
from ._base import generate_categorical_2darray
from ._base import load_evaluation_criteria_raw_values
from ._base import load_sustainability_attributes4electricity_generation_tech
from ._base import load_microclimate_in_office_rooms
from ._base import load_jisperveld_data

from ._dataset_info import KITTI_info
from ._dataset_info import KITTI_info_gap
from ._rs_image import Sentinel2_bandFNs
from ._kml_info import kml_coordiExtraction
from ._img_info import img_exif_info

from ._files_downloading import files_downloading
from ._files_downloading import cifar10_downloading2fixedParams_loader

__all__ = [
    "load_sales_data_cartoon_database",
    "load_ramen_price_cartoon_statistic",
    "load_bowling_contest_cartoon_statistic",
    "load_test_score_cartoon_statistic",
    "baiduPOI_dataCrawler",
    "baiduPOI_dataCrawler_circle",
    "baiduPOI_batchCrawler",
    "KITTI_info",
    "KITTI_info_gap",
    "Sentinel2_bandFNs",
    "kml_coordiExtraction",
    "img_exif_info",
    "generate_categorical_2darray",
    "load_evaluation_criteria_raw_values",
    "load_sustainability_attributes4electricity_generation_tech",
    "load_microclimate_in_office_rooms",
    'load_jisperveld_data',
    'files_downloading',
    'cifar10_downloading2fixedParams_loader',
    ]


