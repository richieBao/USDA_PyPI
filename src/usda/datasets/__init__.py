# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from ._base import load_sales_data_cartoon_database
from ._base import load_ramen_price_cartoon_statistic
from ._base import load_bowling_contest_cartoon_statistic
from ._base import load_test_score_cartoon_statistic
from ._artificial_data import sales_data_cartoon_database

from ._base import baiduPOI_dataCrawler
from ._base import baiduPOI_dataCrawler_circle
from ._base import baiduPOI_batchCrawler
from ._dataset_info import KITTI_info

__all__ = [
    "load_sales_data_cartoon_database",
    "load_ramen_price_cartoon_statistic",
    "load_bowling_contest_cartoon_statistic",
    "load_test_score_cartoon_statistic",
    "sales_data_cartoon_database",
    "baiduPOI_dataCrawler",
    "baiduPOI_dataCrawler_circle",
    "baiduPOI_batchCrawler",
    "KITTI_info"
    ]


