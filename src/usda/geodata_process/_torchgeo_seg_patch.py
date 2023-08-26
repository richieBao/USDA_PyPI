# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:22:22 2023

@author: richie bao
"""
from torchgeo.samplers import GridGeoSampler,RandomGeoSampler,PreChippedGeoSampler
from torchgeo.datasets import stack_samples 
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import matplotlib
import numpy as np
import os
import tarfile
from typing import Optional
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor
from torchgeo.datasets import RasterDataset
from torchvision.transforms import Compose
from torchgeo.transforms import indices,AugmentationSequential

from torchgeo.datasets import GeoDataset
from rasterio.crs import CRS
import functools
from rasterio.io import DatasetReader
import re
import glob
import rasterio
from rasterio.vrt import WarpedVRT
import sys
from torchgeo.trainers import SemanticSegmentationTask
import torch.nn.functional as F
import rioxarray as rxr
import random
import string

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
    Callable,
)

# import _torchgeo_datasets_utils as utils

if __package__:
    from ._torchgeo_datasets_utils import disambiguate_timestamp
    from ._torchgeo_datasets_utils import BoundingBox
else:
    from _torchgeo_datasets_utils import disambiguate_timestamp
    from _torchgeo_datasets_utils import BoundingBox


_EPSILON = 1e-10

class AppendNormalizedDifferenceIndex(IntensityAugmentationBase2D):
    r"""Append normalized difference index as channel to image tensor.

    Computes the following index:

    .. math::

       \text{NDI} = \frac{A - B}{A + B}

    .. versionadded:: 0.2
    """

    def __init__(self, index_a: int, index_b: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_a: reference band channel index
            index_b: difference band channel index
        """
        super().__init__(p=1)
        self.flags = {"index_a": index_a, "index_b": index_b}

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        band_a = input[..., flags["index_a"], :, :]
        band_b = input[..., flags["index_b"], :, :]
        ndi = (band_a - band_b) / (band_a + band_b + _EPSILON)
        ndi = torch.unsqueeze(ndi, -3)
        input = torch.cat((input, ndi), dim=-3)
        return input[0]

class AppendNDVI(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Vegetation Index (NDVI).

    Computes the following index:

    .. math::

       \text{NDVI} = \frac{\text{NIR} - \text{R}}{\text{NIR} + \text{R}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1016/0034-4257(79)90013-0
    """

    def __init__(self, index_nir: int, index_red: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_nir: index of the Near Infrared (NIR) band in the image
            index_red: index of the Red band in the image
        """
        super().__init__(index_a=index_nir, index_b=index_red)

class AppendNDWI(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Water Index (NDWI).

    Computes the following index:

    .. math::

       \text{NDWI} = \frac{\text{G} - \text{NIR}}{\text{G} + \text{NIR}}

    If you use this index in your research, please cite the following paper:

    * https://doi.org/10.1080/01431169608948714
    """

    def __init__(self, index_green: int, index_nir: int) -> None:
        """Initialize a new transform instance.

        Args:
            index_green: index of the Green band in the image
            index_nir: index of the Near Infrared (NIR) band in the image
        """
        super().__init__(index_a=index_green, index_b=index_nir)
        
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
    
#------------------------------------------------------------------------------

class RasterDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as raster files."""

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    filename_glob = "*"

    #: Regular expression used to extract date from filename.
    #:
    #: The expression should use named groups. The expression may contain any number of
    #: groups. The following groups are specifically searched for by the base class:
    #:
    #: * ``date``: used to calculate ``mint`` and ``maxt`` for ``index`` insertion
    #:
    #: When :attr:`~RasterDataset.separate_files` is True, the following additional
    #: groups are searched for to find other files:
    #:
    #: * ``band``: replaced with requested band name
    filename_regex = ".*"

    #: Date format string used to parse date from filename.
    #:
    #: Not used if :attr:`filename_regex` does not contain a ``date`` group.
    date_format = "%Y%m%d"

    #: True if dataset contains imagery, False if dataset contains mask
    is_image = True

    #: True if data is stored in a separate file for each band, else False.
    separate_files = False

    #: Names of all available bands in the dataset
    all_bands: list[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: list[str] = []

    #: Color map for the dataset, used for plotting
    cmap: dict[int, tuple[int, int, int, int]] = {}

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the dataset (overrides the dtype of the data file via a cast).

        Returns:
            the dtype of the dataset

        .. versionadded:: 5.0
        """
        if self.is_image:
            return torch.float32
        else:
            return torch.long

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(transforms)

        self.root = root
        self.bands = bands or self.all_bands
        self.cache = cache
        
        self.filepath_lst=[]
        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:
                        # See if file has a color map
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)
                            except ValueError:
                                pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                            
                        # print(filepath)
                        self.filepath_lst.append(filepath)
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            msg = f"No {self.__class__.__name__} data was found in `root='{self.root}'`"
            if self.bands:
                msg += f" with `bands={self.bands}`"
            raise FileNotFoundError(msg)

        if not self.separate_files:
            self.band_indexes = None
            if self.bands:
                if self.all_bands:
                    self.band_indexes = [
                        self.all_bands.index(i) + 1 for i in self.bands
                    ]
                else:
                    msg = (
                        f"{self.__class__.__name__} is missing an `all_bands` "
                        "attribute, so `bands` cannot be specified."
                    )
                    raise AssertionError(msg)

        self._crs = cast(CRS, crs)
        self._res = cast(float, res)
        self.res=self._res

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        if self.separate_files:
            data_list: list[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if "band" in match.groupdict():
                            start = match.start("band")
                            end = match.end("band")
                            filename = filename[:start] + band + filename[end:]
                    filepath = os.path.join(directory, filename)
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            data = torch.cat(data_list)
        else:
            data = self._merge_files(filepaths, query, self.band_indexes)

        sample = {"crs": self.crs, "bbox": query}

        data = data.to(self.dtype)
        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(
        self,
        filepaths: Sequence[str],
        query: BoundingBox,
        band_indexes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            band_indexes: indexes of bands to be used

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        dest, _ = rasterio.merge.merge(vrt_fhs, bounds, self.res, indexes=band_indexes)

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src        

class NAIP(RasterDataset):
    """National Agriculture Imagery Program (NAIP) dataset.

    The `National Agriculture Imagery Program (NAIP)
    <https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip>`_
    acquires aerial imagery during the agricultural growing seasons in the continental
    U.S. A primary goal of the NAIP program is to make digital ortho photography
    available to governmental agencies and the public within a year of acquisition.

    NAIP is administered by the USDA's Farm Service Agency (FSA) through the Aerial
    Photography Field Office in Salt Lake City. This "leaf-on" imagery is used as a base
    layer for GIS programs in FSA's County Service Centers, and is used to maintain the
    Common Land Unit (CLU) boundaries.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.fisheries.noaa.gov/inport/item/49508/citation
    """

    # https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs141p2_015644.pdf
    # https://planetarycomputer.microsoft.com/dataset/naip#Storage-Documentation
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

    # Plotting
    all_bands = ["R", "G", "B", "NIR"]
    rgb_bands = ["R", "G", "B"]

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        image = sample["image"][0:3, :, :].permute(1, 2, 0)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig          
    
def cmap4LC(LC_color_dict,selected_keys):
    LC_color_dict_selection={k:LC_color_dict[k] for k in selected_keys}
    LC_color_dict_selection[999]=(0,0,0,0)
    cmap_LC, norm=matplotlib.colors.from_levels_and_colors(list(LC_color_dict_selection.keys()),[[v/255 for v in i] for i in LC_color_dict_selection.values()],extend='max')
    return cmap_LC,norm

class Seg_config:       
    # aux_params=dict(
    #     pooling='avg',             # one of 'avg', 'max'
    #     dropout=0.5,               # dropout ratio, default is None
    #     activation='sigmoid',      # activation function, default is None
    #     )
    
    # task=SemanticSegmentationTask(
    #     model='unet',
    #     backbone='resnet34',
    #     weights='imagenet',
    #     pretrained=True,
    #     in_channels=5, 
    #     num_classes=8,
    #     ignore_index=0,
    #     loss='ce', # 'jaccard'
    #     learning_rate=0.1,
    #     learning_rate_schedule_patience=5,
    #     aux_params=aux_params,
    #     )    
    
    LC_color_dict={
        0: (0, 0, 0, 0),
        1: (30, 136, 229,255),
        2: (230, 238, 156, 255),
        3: (46, 125, 50,255),
        4: (205, 220, 57, 255),
        5: (176, 190, 197, 255),
        6: (234, 234, 234, 255),
        7: (189, 189, 189, 255),
        }    
    
def img_size_expand_topNright(X,base=32,w_dim=3,h_dim=2):
    w=X.shape[w_dim]
    h=X.shape[h_dim]
    pad_ex=(0, (w//base)*base+base-w, 0, (h//base)*base+base-h)
    X_=F.pad(input=X, pad=pad_ex, mode='constant', value=0)
       
    return X_,pad_ex    

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    #print("Random string of length", length, "is:", result_str)
    return result_str

def segarray2tiff(seg_array,img_fn,seg_save_dir,seg_name=None,prefix='seg'):
    img=rxr.open_rasterio(img_fn)
    dim1,dim2,dim3=img.data.shape
    da=img.sel(band=1)
    da.data=seg_array[0][:dim2,:dim3]
    
    if seg_name:        
        da.rio.to_raster(os.path.join(seg_save_dir,seg_name),COMPRESS='LZW')
    else:
        da.rio.to_raster(os.path.join(seg_save_dir,f'{prefix}_{os.path.basename(img_fn)}'),COMPRESS='LZW')