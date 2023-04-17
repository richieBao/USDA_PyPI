# -*- coding: utf-8 -*-
if __package__:
    from ._base_options import cfg as cfg_base
    from ._train_options import cfg as cfg_train    
    from ._test_options import cfg as cfg_test
else:
    from _base_options import cfg as cfg_base
    from _train_options import cfg as cfg_train   
    from _test_options import cfg as cfg_test
   

__all__ = [
    'cfg_base',
    'cfg_train',
    "cfg_test"
    ]

