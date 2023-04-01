# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 09:25:17 2023

@author: richie bao
"""
from torchgeo.transforms import indices
from torchgeo.datasets import RasterDataset
import kornia.augmentation as K
from torch import Tensor

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

def remove_bbox(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Removes the bounding box property from a sample.

    Args:
        sample: dictionary with geographic metadata

    Returns
        sample without the bbox property
    """
    del sample["bbox"]
    return sample

def naip_preprocess(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single sample from the NAIP Dataset.

    Args:
        sample: NAIP image dictionary

    Returns:
        preprocessed NAIP data
    """
    sample["image"] = sample["image"].float()
    sample["image"] /= 255.0

    return sample

class Naip_dividedby255(K.IntensityAugmentationBase2D):
    def __init__(self):
        super().__init__(p=1)
        self.flags = {'denominator':255}
    
    def apply_transform(self,input: Tensor,params: Dict[str, Tensor],flags: Dict[str, int],transform: Optional[Tensor] = None,) -> Tensor:
        input_f=input.float()
        input_f/=flags['denominator']
        return input_f

class naip_rd(RasterDataset):
    filename_glob = "m_*.*"
    filename_regex = r"""
        ^m
        _(?P<quadrangle>\d+)
        _(?P<quarter_quad>[a-z]+)
        _(?P<utm_zone>\d+)
        _(?P<resolution>\d+)
        _(?P<date>\d+)
        (?:_(?P<processing_date>\d+))?
        \..*$
    """
    is_image=True
    
    all_bands = ["R", "G", "B", "NIR"]
    rgb_bands = ["R", "G", "B"]    
    


