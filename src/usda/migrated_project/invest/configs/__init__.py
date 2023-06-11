# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:11:00 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
if __package__:
    # from ._carbon_configs import cfg as cfg_carbon
    from . import _carbon_configs as cfg_carbon
    from . import _habitat_quality_configs as cfg_habitat_quality
    from . import _crop_pollination_configs as cfg_crop_pollination
    from . import _crop_production_percentile_configs as cfg_crop_production_percentile
    from . import _crop_production_regression_configs as cfg_crop_production_regression
    from . import _urban_cooling_configs as cfg_urban_cooling
else:
    import _carbon_configs as cfg_carbon
    import _habitat_quality_configs as cfg_habitat_quality
    import _crop_pollination_configs as cfg_crop_pollination
    import _crop_production_percentile_configs as cfg_crop_production_percentile
    import _crop_production_regression_configs as cfg_crop_production_regression
    import _urban_cooling_configs as cfg_urban_cooling

__all__ = [
    'cfg_carbon',
    'cfg_habitat_quality',
    'cfg_crop_pollination',
    'cfg_crop_production_regression',
    ]

