# -*- coding: utf-8 -*-
"""Pollinator service model for InVEST."""
import collections
import hashlib
import inspect
import itertools
import logging
import os
import re

import numpy
import pygeoprocessing
import taskgraph
from osgeo import gdal
from osgeo import ogr

import sys,os
sys.path.append('..')

if __package__:
    from . import _utils as utils
    from ..configs import cfg_crop_pollination
else:
    import _utils as utils
    from configs import cfg_crop_pollination
    
LOGGER = logging.getLogger(__name__)

_INDEX_NODATA = cfg_crop_pollination._INDEX_NODATA
_NESTING_SUBSTRATE_PATTERN = cfg_crop_pollination._NESTING_SUBSTRATE_PATTERN
_FLORAL_RESOURCES_AVAILABLE_PATTERN = cfg_crop_pollination._FLORAL_RESOURCES_AVAILABLE_PATTERN
_EXPECTED_BIOPHYSICAL_HEADERS = cfg_crop_pollination._EXPECTED_BIOPHYSICAL_HEADERS
_NESTING_SUITABILITY_PATTERN = cfg_crop_pollination._NESTING_SUITABILITY_PATTERN
_FORAGING_ACTIVITY_PATTERN = cfg_crop_pollination._FORAGING_ACTIVITY_PATTERN
_FORAGING_ACTIVITY_RE_PATTERN = cfg_crop_pollination._FORAGING_ACTIVITY_RE_PATTERN
_RELATIVE_SPECIES_ABUNDANCE_FIELD = cfg_crop_pollination._RELATIVE_SPECIES_ABUNDANCE_FIELD
_ALPHA_HEADER = cfg_crop_pollination._ALPHA_HEADER
_EXPECTED_GUILD_HEADERS = cfg_crop_pollination._EXPECTED_GUILD_HEADERS
_NESTING_SUBSTRATE_INDEX_FILEPATTERN = cfg_crop_pollination._NESTING_SUBSTRATE_INDEX_FILEPATTERN
_FARM_NESTING_SUBSTRATE_INDEX_FILEPATTERN = cfg_crop_pollination._FARM_NESTING_SUBSTRATE_INDEX_FILEPATTERN
_HABITAT_NESTING_INDEX_FILE_PATTERN = cfg_crop_pollination._HABITAT_NESTING_INDEX_FILE_PATTERN
_RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN = cfg_crop_pollination._RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN
_FARM_RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN = cfg_crop_pollination._FARM_RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN
_LOCAL_FORAGING_EFFECTIVENESS_FILE_PATTERN = cfg_crop_pollination._LOCAL_FORAGING_EFFECTIVENESS_FILE_PATTERN
_FLORAL_RESOURCES_INDEX_FILE_PATTERN = cfg_crop_pollination._FLORAL_RESOURCES_INDEX_FILE_PATTERN
_POLLINATOR_SUPPLY_FILE_PATTERN = cfg_crop_pollination._POLLINATOR_SUPPLY_FILE_PATTERN
_PROJECTED_FARM_VECTOR_FILE_PATTERN = cfg_crop_pollination._PROJECTED_FARM_VECTOR_FILE_PATTERN
_KERNEL_FILE_PATTERN = cfg_crop_pollination._KERNEL_FILE_PATTERN
_POLLINATOR_ABUNDANCE_FILE_PATTERN = cfg_crop_pollination._POLLINATOR_ABUNDANCE_FILE_PATTERN
_TOTAL_POLLINATOR_ABUNDANCE_FILE_PATTERN = cfg_crop_pollination._TOTAL_POLLINATOR_ABUNDANCE_FILE_PATTERN
_FORAGED_FLOWERS_INDEX_FILE_PATTERN = cfg_crop_pollination._FORAGED_FLOWERS_INDEX_FILE_PATTERN
_CONVOLVE_PS_FILE_PATH = cfg_crop_pollination._CONVOLVE_PS_FILE_PATH
_HALF_SATURATION_FILE_PATTERN = cfg_crop_pollination._HALF_SATURATION_FILE_PATTERN
_BLANK_RASTER_FILE_PATTERN = cfg_crop_pollination._BLANK_RASTER_FILE_PATTERN
_FARM_POLLINATOR_SEASON_FILE_PATTERN = cfg_crop_pollination._FARM_POLLINATOR_SEASON_FILE_PATTERN
_FARM_POLLINATOR_FILE_PATTERN = cfg_crop_pollination._FARM_POLLINATOR_FILE_PATTERN
_MANAGED_POLLINATOR_FILE_PATTERN = cfg_crop_pollination._MANAGED_POLLINATOR_FILE_PATTERN
_TOTAL_POLLINATOR_YIELD_FILE_PATTERN = cfg_crop_pollination._TOTAL_POLLINATOR_YIELD_FILE_PATTERN
_WILD_POLLINATOR_YIELD_FILE_PATTERN = cfg_crop_pollination._WILD_POLLINATOR_YIELD_FILE_PATTERN
_FARM_VECTOR_RESULT_FILE_PATTERN = cfg_crop_pollination._FARM_VECTOR_RESULT_FILE_PATTERN
_TOTAL_FARM_YIELD_FIELD_ID = cfg_crop_pollination._TOTAL_FARM_YIELD_FIELD_ID
_WILD_POLLINATOR_FARM_YIELD_FIELD_ID = cfg_crop_pollination._WILD_POLLINATOR_FARM_YIELD_FIELD_ID
_POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID = cfg_crop_pollination._POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID
_POLLINATOR_ABUNDANCE_FARM_FIELD_ID = cfg_crop_pollination._POLLINATOR_ABUNDANCE_FARM_FIELD_ID
_FARM_FLORAL_RESOURCES_HEADER_PATTERN = cfg_crop_pollination._FARM_FLORAL_RESOURCES_HEADER_PATTERN
_FARM_FLORAL_RESOURCES_PATTERN = cfg_crop_pollination._FARM_FLORAL_RESOURCES_PATTERN
_FARM_NESTING_SUBSTRATE_HEADER_PATTERN = cfg_crop_pollination._FARM_NESTING_SUBSTRATE_HEADER_PATTERN
_FARM_NESTING_SUBSTRATE_RE_PATTERN = cfg_crop_pollination._FARM_NESTING_SUBSTRATE_RE_PATTERN
_HALF_SATURATION_FARM_HEADER = cfg_crop_pollination._HALF_SATURATION_FARM_HEADER
_CROP_POLLINATOR_DEPENDENCE_FIELD = cfg_crop_pollination._CROP_POLLINATOR_DEPENDENCE_FIELD
_MANAGED_POLLINATORS_FIELD = cfg_crop_pollination._MANAGED_POLLINATORS_FIELD
_FARM_SEASON_FIELD = cfg_crop_pollination._FARM_SEASON_FIELD
_EXPECTED_FARM_HEADERS = cfg_crop_pollination._EXPECTED_FARM_HEADERS 

class Crop_pollination:
    """Pollination.

    Args:
        args['workspace_dir'] (string): a path to the output workspace folder.
            Will overwrite any files that exist if the path already exists.
        args['results_suffix'] (string): string appended to each output
            file path.
        args['landcover_raster_path'] (string): file path to a landcover
            raster.
        args['guild_table_path'] (string): file path to a table indicating
            the bee species to analyze in this model run.  Table headers
            must include:

                * 'species': a bee species whose column string names will
                    be referred to in other tables and the model will output
                    analyses per species.
                * one or more columns matching _NESTING_SUITABILITY_PATTERN
                    with values in the range [0, 1] indicating the
                    suitability of the given species to nest in a particular
                    substrate.
                * one or more columns matching _FORAGING_ACTIVITY_RE_PATTERN
                    with values in the range [0, 1] indicating the
                    relative level of foraging activity for that species
                    during a particular season.
                * _ALPHA_HEADER the sigma average flight distance of that bee
                    species in meters.
                * 'relative_abundance': a weight indicating the relative
                    abundance of the particular species with respect to the
                    sum of all relative abundance weights in the table.

        args['landcover_biophysical_table_path'] (string): path to a table
            mapping landcover codes in `args['landcover_path']` to indexes of
            nesting availability for each nesting substrate referenced in
            guilds table as well as indexes of abundance of floral resources
            on that landcover type per season in the bee activity columns of
            the guild table.

            All indexes are in the range [0, 1].

            Columns in the table must be at least
                * 'lucode': representing all the unique landcover codes in
                    the raster ast `args['landcover_path']`
                * For every nesting matching _NESTING_SUITABILITY_PATTERN
                  in the guild stable, a column matching the pattern in
                  `_LANDCOVER_NESTING_INDEX_HEADER`.
                * For every season matching _FORAGING_ACTIVITY_RE_PATTERN
                  in the guilds table, a column matching
                  the pattern in `_LANDCOVER_FLORAL_RESOURCES_INDEX_HEADER`.
        args['farm_vector_path'] (string): (optional) path to a single layer
            polygon shapefile representing farms. If present will trigger the
            farm yield component of the model.

            The layer must have at least the following fields:

            * season (string): season in which the farm needs pollination
            * crop_type (string): a text field to identify the crop type for
                summary statistics.
            * half_sat (float): a real in the range [0, 1] representing
                the proportion of wild pollinators to achieve a 50% yield
                of that crop.
            * p_dep (float): a number in the range [0, 1]
                representing the proportion of yield dependent on pollinators.
            * p_managed (float): proportion of pollinators that come from
                non-native/managed hives.
            * fr_[season] (float): one or more fields that match this pattern
                such that `season` also matches the season headers in the
                biophysical and guild table.  Any areas that overlap the
                landcover map will replace seasonal floral resources with
                this value.  Ranges from 0..1.
            * n_[substrate] (float): One or more fields that match this
                pattern such that `substrate` also matches the nesting
                substrate headers in the biophysical and guild table.  Any
                areas that overlap the landcover map will replace nesting
                substrate suitability with this value.  Ranges from 0..1.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None
    """        
    
    def __init__(self,args):
        paths_dict={'target_farm_result_path':[],
					 'wild_pollinator_yield_path':[],
					 'total_pollinator_yield_path':[],
					 'managed_pollinator_path':[],
					 'farm_pollinator_path':[],
					 'farm_pollinator_season_path':[],
					 'half_saturation_raster_path':[],
					 'total_pollinator_abundance_index_path':[],
					 'pollinator_abundance_path':[]}
        
       # create initial working directories and determine file suffixes
        intermediate_output_dir = os.path.join(
            args['workspace_dir'], 'intermediate_outputs')
        work_token_dir = os.path.join(
            intermediate_output_dir, '_taskgraph_working_dir')
        output_dir = os.path.join(args['workspace_dir'])
        utils.make_directories(
            [output_dir, intermediate_output_dir])
        file_suffix = utils.make_suffix_string(args, 'results_suffix')
    
        if 'farm_vector_path' in args and args['farm_vector_path'] != '':
            # we set the vector path to be the projected vector that we'll create
            # later
            farm_vector_path = os.path.join(
                intermediate_output_dir,
                _PROJECTED_FARM_VECTOR_FILE_PATTERN % file_suffix)
        else:
            farm_vector_path = None
    
        # parse out the scenario variables from a complicated set of two tables
        # and possibly a farm polygon.  This function will also raise an exception
        # if any of the inputs are malformed.
        scenario_variables = self._parse_scenario_variables(args)
        landcover_raster_info = pygeoprocessing.get_raster_info(
            args['landcover_raster_path'])
    
        try:
            n_workers = int(args['n_workers'])
        except (KeyError, ValueError, TypeError):
            # KeyError when n_workers is not present in args
            # ValueError when n_workers is an empty string.
            # TypeError when n_workers is None.
            n_workers = -1  # Synchronous mode.
        task_graph = taskgraph.TaskGraph(work_token_dir, n_workers)
    
        if farm_vector_path is not None:
            # ensure farm vector is in the same projection as the landcover map
            reproject_farm_task = task_graph.add_task(
                task_name='reproject_farm_task',
                func=pygeoprocessing.reproject_vector,
                args=(
                    args['farm_vector_path'],
                    landcover_raster_info['projection_wkt'], farm_vector_path),
                target_path_list=[farm_vector_path])
    
        # calculate nesting_substrate_index[substrate] substrate maps
        # N(x, n) = ln(l(x), n)
        scenario_variables['nesting_substrate_index_path'] = {}
        landcover_substrate_index_tasks = {}
        reclass_error_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Biophysical'}
        for substrate in scenario_variables['substrate_list']:
            nesting_substrate_index_path = os.path.join(
                intermediate_output_dir,
                _NESTING_SUBSTRATE_INDEX_FILEPATTERN % (substrate, file_suffix))
            scenario_variables['nesting_substrate_index_path'][substrate] = (
                nesting_substrate_index_path)
    
            landcover_substrate_index_tasks[substrate] = task_graph.add_task(
                task_name=f'reclassify_to_substrate_{substrate}',
                func=utils.reclassify_raster,
                args=(
                    (args['landcover_raster_path'], 1),
                    scenario_variables['landcover_substrate_index'][substrate],
                    nesting_substrate_index_path, gdal.GDT_Float32,
                    _INDEX_NODATA, reclass_error_details),
                target_path_list=[nesting_substrate_index_path])
    
        # calculate farm_nesting_substrate_index[substrate] substrate maps
        # dependent on farm substrate rasterized over N(x, n)
        if farm_vector_path is not None:
            scenario_variables['farm_nesting_substrate_index_path'] = (
                collections.defaultdict(dict))
            farm_substrate_rasterize_task_list = []
            for substrate in scenario_variables['substrate_list']:
                farm_substrate_id = (
                    _FARM_NESTING_SUBSTRATE_HEADER_PATTERN % substrate)
                farm_nesting_substrate_index_path = os.path.join(
                    intermediate_output_dir,
                    _FARM_NESTING_SUBSTRATE_INDEX_FILEPATTERN % (
                        substrate, file_suffix))
                scenario_variables['farm_nesting_substrate_index_path'][
                    substrate] = farm_nesting_substrate_index_path
                farm_substrate_rasterize_task_list.append(
                    task_graph.add_task(
                        task_name=f'rasterize_nesting_substrate_{substrate}',
                        func=self._rasterize_vector_onto_base,
                        args=(
                            scenario_variables['nesting_substrate_index_path'][
                                substrate],
                            farm_vector_path, farm_substrate_id,
                            farm_nesting_substrate_index_path),
                        target_path_list=[farm_nesting_substrate_index_path],
                        dependent_task_list=[
                            landcover_substrate_index_tasks[substrate],
                            reproject_farm_task]))
    
        habitat_nesting_tasks = {}
        scenario_variables['habitat_nesting_index_path'] = {}
        for species in scenario_variables['species_list']:
            # calculate habitat_nesting_index[species] HN(x, s) = max_n(N(x, n) ns(s,n))
            if farm_vector_path is not None:
                dependent_task_list = farm_substrate_rasterize_task_list
                substrate_path_map = scenario_variables[
                    'farm_nesting_substrate_index_path']
            else:
                dependent_task_list = landcover_substrate_index_tasks.values()
                substrate_path_map = scenario_variables[
                    'nesting_substrate_index_path']
    
            scenario_variables['habitat_nesting_index_path'][species] = (
                os.path.join(
                    intermediate_output_dir,
                    _HABITAT_NESTING_INDEX_FILE_PATTERN % (species, file_suffix)))
    
            calculate_habitat_nesting_index_op = _CalculateHabitatNestingIndex(
                substrate_path_map,
                scenario_variables['species_substrate_index'][species],
                scenario_variables['habitat_nesting_index_path'][species])
    
            habitat_nesting_tasks[species] = task_graph.add_task(
                task_name=f'calculate_habitat_nesting_{species}',
                func=calculate_habitat_nesting_index_op,
                dependent_task_list=dependent_task_list,
                target_path_list=[
                    scenario_variables['habitat_nesting_index_path'][species]])
    
        scenario_variables['relative_floral_abundance_index_path'] = {}
        relative_floral_abudance_task_map = {}
        reclass_error_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Biophysical'}
        for season in scenario_variables['season_list']:
            # calculate relative_floral_abundance_index[season] per season
            # RA(l(x), j)
            relative_floral_abundance_index_path = os.path.join(
                intermediate_output_dir,
                _RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN % (
                    season, file_suffix))
    
            relative_floral_abudance_task = task_graph.add_task(
                task_name=f'reclassify_to_floral_abundance_{season}',
                func=utils.reclassify_raster,
                args=(
                    (args['landcover_raster_path'], 1),
                    scenario_variables['landcover_floral_resources'][season],
                    relative_floral_abundance_index_path, gdal.GDT_Float32,
                    _INDEX_NODATA, reclass_error_details),
                target_path_list=[relative_floral_abundance_index_path])
    
            # if there's a farm, rasterize floral resources over the top
            if farm_vector_path is not None:
                farm_relative_floral_abundance_index_path = os.path.join(
                    intermediate_output_dir,
                    _FARM_RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN % (
                        season, file_suffix))
    
                # this is the shapefile header for the farm seasonal floral
                # resources
                farm_floral_resources_id = (
                    _FARM_FLORAL_RESOURCES_HEADER_PATTERN % season)
    
                # override the relative floral task because we'll need this one
                relative_floral_abudance_task = task_graph.add_task(
                    task_name=f'relative_floral_abudance_task_{season}',
                    func=self._rasterize_vector_onto_base,
                    args=(
                        relative_floral_abundance_index_path,
                        farm_vector_path, farm_floral_resources_id,
                        farm_relative_floral_abundance_index_path),
                    target_path_list=[
                        farm_relative_floral_abundance_index_path],
                    dependent_task_list=[
                        relative_floral_abudance_task, reproject_farm_task])
    
                # override the relative floral abundance index path since we'll
                # need the farm one
                relative_floral_abundance_index_path = (
                    farm_relative_floral_abundance_index_path)
    
            scenario_variables['relative_floral_abundance_index_path'][season] = (
                relative_floral_abundance_index_path)
            relative_floral_abudance_task_map[season] = (
                relative_floral_abudance_task)
    
        scenario_variables['foraged_flowers_index_path'] = {}
        foraged_flowers_index_task_map = {}
        for species in scenario_variables['species_list']:
            for season in scenario_variables['season_list']:
                # calculate foraged_flowers_species_season = RA(l(x),j)*fa(s,j)
                foraged_flowers_index_path = os.path.join(
                    intermediate_output_dir,
                    _FORAGED_FLOWERS_INDEX_FILE_PATTERN % (
                        species, season, file_suffix))
                relative_abundance_path = (
                    scenario_variables['relative_floral_abundance_index_path'][
                        season])
                mult_by_scalar_op = _MultByScalar(
                    scenario_variables['species_foraging_activity'][
                        (species, season)])
                foraged_flowers_index_task_map[(species, season)] = (
                    task_graph.add_task(
                        task_name=f'calculate_foraged_flowers_{species}_{season}',
                        func=pygeoprocessing.raster_calculator,
                        args=(
                            [(relative_abundance_path, 1)],
                            mult_by_scalar_op, foraged_flowers_index_path,
                            gdal.GDT_Float32, _INDEX_NODATA),
                        dependent_task_list=[
                            relative_floral_abudance_task_map[season]],
                        target_path_list=[foraged_flowers_index_path]))
                scenario_variables['foraged_flowers_index_path'][
                    (species, season)] = foraged_flowers_index_path
    
        pollinator_abundance_path_map = {}
        pollinator_abundance_task_map = {}
        floral_resources_index_path_map = {}
        floral_resources_index_task_map = {}
        for species in scenario_variables['species_list']:
            # calculate foraging_effectiveness[species]
            # FE(x, s) = sum_j [RA(l(x), j) * fa(s, j)]
            foraged_flowers_path_band_list = [
                (scenario_variables['foraged_flowers_index_path'][
                    (species, season)], 1)
                for season in scenario_variables['season_list']]
            local_foraging_effectiveness_path = os.path.join(
                intermediate_output_dir,
                _LOCAL_FORAGING_EFFECTIVENESS_FILE_PATTERN % (
                    species, file_suffix))
    
            local_foraging_effectiveness_task = task_graph.add_task(
                task_name=f'local_foraging_effectiveness_{species}',
                func=pygeoprocessing.raster_calculator,
                args=(
                    foraged_flowers_path_band_list,
                    _SumRasters(), local_foraging_effectiveness_path,
                    gdal.GDT_Float32, _INDEX_NODATA),
                target_path_list=[
                    local_foraging_effectiveness_path],
                dependent_task_list=[
                    foraged_flowers_index_task_map[(species, season)]
                    for season in scenario_variables['season_list']])
    
            landcover_pixel_size_tuple = landcover_raster_info['pixel_size']
            try:
                landcover_mean_pixel_size = utils.mean_pixel_size_and_area(
                    landcover_pixel_size_tuple)[0]
            except ValueError:
                landcover_mean_pixel_size = numpy.min(numpy.absolute(
                    landcover_pixel_size_tuple))
                LOGGER.debug(
                    'Land Cover Raster has unequal x, y pixel sizes: '
                    f'{landcover_pixel_size_tuple}. Using'
                    f'{landcover_mean_pixel_size} as the mean pixel size.')
            # create a convolution kernel for the species flight range
            alpha = (
                scenario_variables['alpha_value'][species] /
                landcover_mean_pixel_size)
            kernel_path = os.path.join(
                intermediate_output_dir, _KERNEL_FILE_PATTERN % (
                    alpha, file_suffix))
    
            alpha_kernel_raster_task = task_graph.add_task(
                task_name=f'decay_kernel_raster_{alpha}',
                func=utils.exponential_decay_kernel_raster,
                args=(alpha, kernel_path),
                target_path_list=[kernel_path])
    
            # convolve FE with alpha_s
            floral_resources_index_path = os.path.join(
                intermediate_output_dir, _FLORAL_RESOURCES_INDEX_FILE_PATTERN % (
                    species, file_suffix))
            floral_resources_index_path_map[species] = floral_resources_index_path
    
            floral_resources_task = task_graph.add_task(
                task_name=f'convolve_{species}',
                func=pygeoprocessing.convolve_2d,
                args=(
                    (local_foraging_effectiveness_path, 1), (kernel_path, 1),
                    floral_resources_index_path),
                kwargs={
                    'ignore_nodata_and_edges': True,
                    'mask_nodata': True,
                    'normalize_kernel': False,
                    },
                dependent_task_list=[
                    alpha_kernel_raster_task, local_foraging_effectiveness_task],
                target_path_list=[floral_resources_index_path])
    
            floral_resources_index_task_map[species] = floral_resources_task
            # calculate
            # pollinator_supply_index[species] PS(x,s) = FR(x,s) * HN(x,s) * sa(s)
            pollinator_supply_index_path = os.path.join(
                output_dir, _POLLINATOR_SUPPLY_FILE_PATTERN % (
                    species, file_suffix))
            ps_index_op = _PollinatorSupplyIndexOp(
                scenario_variables['species_abundance'][species])
            pollinator_supply_task = task_graph.add_task(
                task_name=f'calculate_pollinator_supply_{species}',
                func=pygeoprocessing.raster_calculator,
                args=(
                    [(scenario_variables['habitat_nesting_index_path'][species],
                      1),
                     (floral_resources_index_path, 1)], ps_index_op,
                    pollinator_supply_index_path, gdal.GDT_Float32,
                    _INDEX_NODATA),
                dependent_task_list=[
                    floral_resources_task, habitat_nesting_tasks[species]],
                target_path_list=[pollinator_supply_index_path])
    
            # calc convolved_PS PS over alpha_s
            convolve_ps_path = os.path.join(
                intermediate_output_dir, _CONVOLVE_PS_FILE_PATH % (
                    species, file_suffix))
    
            convolve_ps_task = task_graph.add_task(
                task_name=f'convolve_ps_{species}',
                func=pygeoprocessing.convolve_2d,
                args=(
                    (pollinator_supply_index_path, 1), (kernel_path, 1),
                    convolve_ps_path),
                kwargs={
                    'ignore_nodata_and_edges': True,
                    'mask_nodata': True,
                    'normalize_kernel': False,
                    },
                dependent_task_list=[
                    alpha_kernel_raster_task, pollinator_supply_task],
                target_path_list=[convolve_ps_path])
    
            for season in scenario_variables['season_list']:
                # calculate pollinator activity as
                # PA(x,s,j)=RA(l(x),j)fa(s,j) convolve(ps, alpha_s)
                foraged_flowers_index_path = (
                    scenario_variables['foraged_flowers_index_path'][
                        (species, season)])
                pollinator_abundance_path = os.path.join(
                    output_dir, _POLLINATOR_ABUNDANCE_FILE_PATTERN % (
                        species, season, file_suffix))
                pollinator_abundance_task_map[(species, season)] = (
                    task_graph.add_task(
                        task_name=f'calculate_poll_abudance_{species}',
                        func=pygeoprocessing.raster_calculator,
                        args=(
                            [(foraged_flowers_index_path, 1),
                             (floral_resources_index_path_map[species], 1),
                             (convolve_ps_path, 1)],
                            _PollinatorSupplyOp(), pollinator_abundance_path,
                            gdal.GDT_Float32, _INDEX_NODATA),
                        dependent_task_list=[
                            foraged_flowers_index_task_map[(species, season)],
                            floral_resources_index_task_map[species],
                            convolve_ps_task],
                        target_path_list=[pollinator_abundance_path]))
                pollinator_abundance_path_map[(species, season)] = (
                    pollinator_abundance_path)
                
                paths_dict['pollinator_abundance_path'].append(pollinator_abundance_path)
    
        # calculate total abundance of all pollinators for each season
        total_pollinator_abundance_task = {}
        for season in scenario_variables['season_list']:
            # total_pollinator_abundance_index[season] PAT(x,j)=sum_s PA(x,s,j)
            total_pollinator_abundance_index_path = os.path.join(
                output_dir, _TOTAL_POLLINATOR_ABUNDANCE_FILE_PATTERN % (
                    season, file_suffix))
    
            pollinator_abundance_season_path_band_list = [
                (pollinator_abundance_path_map[(species, season)], 1)
                for species in scenario_variables['species_list']]
    
            total_pollinator_abundance_task[season] = task_graph.add_task(
                task_name=f'calculate_poll_abudnce_{species}_{season}',
                func=pygeoprocessing.raster_calculator,
                args=(
                    pollinator_abundance_season_path_band_list, _SumRasters(),
                    total_pollinator_abundance_index_path, gdal.GDT_Float32,
                    _INDEX_NODATA),
                dependent_task_list=[
                    pollinator_abundance_task_map[(species, season)]
                    for species in scenario_variables['species_list']],
                target_path_list=[total_pollinator_abundance_index_path])
            
            paths_dict['total_pollinator_abundance_index_path'].append(total_pollinator_abundance_index_path)
    
        # next step is farm vector calculation, if no farms then okay to quit
        if farm_vector_path is None:
            task_graph.close()
            task_graph.join()
            return
    
        # blank raster used for rasterizing all the farm parameters/fields later
        blank_raster_path = os.path.join(
            intermediate_output_dir, _BLANK_RASTER_FILE_PATTERN % file_suffix)
        blank_raster_task = task_graph.add_task(
            task_name='create_blank_raster',
            func=pygeoprocessing.new_raster_from_base,
            args=(
                args['landcover_raster_path'], blank_raster_path,
                gdal.GDT_Float32, [_INDEX_NODATA]),
            kwargs={'fill_value_list': [_INDEX_NODATA]},
            target_path_list=[blank_raster_path])
    
        farm_pollinator_season_path_list = []
        farm_pollinator_season_task_list = []
        for season in scenario_variables['season_list']:
    
            half_saturation_raster_path = os.path.join(
                intermediate_output_dir, _HALF_SATURATION_FILE_PATTERN % (
                    season, file_suffix))
            half_saturation_task = task_graph.add_task(
                task_name=f'half_saturation_rasterize_{season}',
                func=self._rasterize_vector_onto_base,
                args=(
                    blank_raster_path, farm_vector_path,
                    _HALF_SATURATION_FARM_HEADER, half_saturation_raster_path),
                kwargs={'filter_string': f"{_FARM_SEASON_FIELD}='{season}'"},
                dependent_task_list=[blank_raster_task],
                target_path_list=[half_saturation_raster_path])
            
            paths_dict['half_saturation_raster_path'].append(half_saturation_raster_path)
    
            # calc on farm pollinator abundance i.e. FP_season
            farm_pollinator_season_path = os.path.join(
                intermediate_output_dir, _FARM_POLLINATOR_SEASON_FILE_PATTERN % (
                    season, file_suffix))
            farm_pollinator_season_task_list.append(task_graph.add_task(
                task_name=f'farm_pollinator_{season}',
                func=pygeoprocessing.raster_calculator,
                args=(
                    [(half_saturation_raster_path, 1),
                     (total_pollinator_abundance_index_path, 1)],
                    _OnFarmPollinatorAbundance(), farm_pollinator_season_path,
                    gdal.GDT_Float32, _INDEX_NODATA),
                dependent_task_list=[
                    half_saturation_task, total_pollinator_abundance_task[season]],
                target_path_list=[farm_pollinator_season_path]))
            farm_pollinator_season_path_list.append(farm_pollinator_season_path)
            
            paths_dict['farm_pollinator_season_path'].append(farm_pollinator_season_path)
    
        # sum farm pollinators
        farm_pollinator_path = os.path.join(
            output_dir, _FARM_POLLINATOR_FILE_PATTERN % file_suffix)
        farm_pollinator_task = task_graph.add_task(
            task_name='sum_farm_pollinators',
            func=pygeoprocessing.raster_calculator,
            args=(
                [(path, 1) for path in farm_pollinator_season_path_list],
                _SumRasters(), farm_pollinator_path, gdal.GDT_Float32,
                _INDEX_NODATA),
            dependent_task_list=farm_pollinator_season_task_list,
            target_path_list=[farm_pollinator_path])
        
        paths_dict['farm_pollinator_path'].append(farm_pollinator_path)
    
        # rasterize managed pollinators
        managed_pollinator_path = os.path.join(
            intermediate_output_dir,
            _MANAGED_POLLINATOR_FILE_PATTERN % file_suffix)
        managed_pollinator_task = task_graph.add_task(
            task_name='rasterize_managed_pollinators',
            func=self._rasterize_vector_onto_base,
            args=(
                blank_raster_path, farm_vector_path, _MANAGED_POLLINATORS_FIELD,
                managed_pollinator_path),
            dependent_task_list=[reproject_farm_task, blank_raster_task],
            target_path_list=[managed_pollinator_path])
        
        paths_dict['managed_pollinator_path'].append(managed_pollinator_path)
    
        # calculate PYT
        total_pollinator_yield_path = os.path.join(
            output_dir, _TOTAL_POLLINATOR_YIELD_FILE_PATTERN % file_suffix)
        pyt_task = task_graph.add_task(
            task_name='calculate_total_pollinators',
            func=pygeoprocessing.raster_calculator,
            args=(
                [(managed_pollinator_path, 1), (farm_pollinator_path, 1)],
                _PYTOp(), total_pollinator_yield_path, gdal.GDT_Float32,
                _INDEX_NODATA),
            dependent_task_list=[farm_pollinator_task, managed_pollinator_task],
            target_path_list=[total_pollinator_yield_path])
        
        paths_dict['total_pollinator_yield_path'].append(total_pollinator_yield_path)
    
        # calculate PYW
        wild_pollinator_yield_path = os.path.join(
            output_dir, _WILD_POLLINATOR_YIELD_FILE_PATTERN % file_suffix)
        wild_pollinator_task = task_graph.add_task(
            task_name='calcualte_wild_pollinators',
            func=pygeoprocessing.raster_calculator,
            args=(
                [(managed_pollinator_path, 1), (total_pollinator_yield_path, 1)],
                _PYWOp(), wild_pollinator_yield_path, gdal.GDT_Float32,
                _INDEX_NODATA),
            dependent_task_list=[pyt_task, managed_pollinator_task],
            target_path_list=[wild_pollinator_yield_path])
        
        paths_dict['wild_pollinator_yield_path'].append(wild_pollinator_yield_path)
    
        # aggregate yields across farms
        target_farm_result_path = os.path.join(
            output_dir, _FARM_VECTOR_RESULT_FILE_PATTERN % file_suffix)
        if os.path.exists(target_farm_result_path):
            os.remove(target_farm_result_path)
        reproject_farm_task.join()
        self._create_farm_result_vector(
            farm_vector_path, target_farm_result_path)
    
        # aggregate wild pollinator yield over farm
        wild_pollinator_task.join()
        wild_pollinator_yield_aggregate = pygeoprocessing.zonal_statistics(
            (wild_pollinator_yield_path, 1), target_farm_result_path)
    
        # aggregate yield over a farm
        pyt_task.join()
        total_farm_results = pygeoprocessing.zonal_statistics(
            (total_pollinator_yield_path, 1), target_farm_result_path)
    
        # aggregate the pollinator abundance results over the farms
        pollinator_abundance_results = {}
        for season in scenario_variables['season_list']:
            total_pollinator_abundance_index_path = os.path.join(
                output_dir, _TOTAL_POLLINATOR_ABUNDANCE_FILE_PATTERN % (
                    season, file_suffix))
            total_pollinator_abundance_task[season].join()
            pollinator_abundance_results[season] = (
                pygeoprocessing.zonal_statistics(
                    (total_pollinator_abundance_index_path, 1),
                    target_farm_result_path))
    
        target_farm_vector = gdal.OpenEx(target_farm_result_path, 1)
        target_farm_layer = target_farm_vector.GetLayer()
        
        paths_dict['target_farm_result_path'].append(target_farm_result_path)
    
        # aggregate results per farm
        for farm_feature in target_farm_layer:
            nu = float(farm_feature.GetField(_CROP_POLLINATOR_DEPENDENCE_FIELD))
            fid = farm_feature.GetFID()
            if total_farm_results[fid]['count'] > 0:
                # total pollinator farm yield is 1-*nu(1-tot_pollination_coverage)
                # this is YT from the user's guide (y_tot)
                farm_feature.SetField(
                    _TOTAL_FARM_YIELD_FIELD_ID,
                    1 - nu * (
                        1 - total_farm_results[fid]['sum'] /
                        float(total_farm_results[fid]['count'])))
    
                # this is PYW ('pdep_y_w')
                farm_feature.SetField(
                    _POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID,
                    (wild_pollinator_yield_aggregate[fid]['sum'] /
                     float(wild_pollinator_yield_aggregate[fid]['count'])))
    
                # this is YW ('y_wild')
                farm_feature.SetField(
                    _WILD_POLLINATOR_FARM_YIELD_FIELD_ID,
                    nu * (wild_pollinator_yield_aggregate[fid]['sum'] /
                          float(wild_pollinator_yield_aggregate[fid]['count'])))
    
                # this is PAT ('p_abund')
                farm_season = farm_feature.GetField(_FARM_SEASON_FIELD)
                farm_feature.SetField(
                    _POLLINATOR_ABUNDANCE_FARM_FIELD_ID,
                    pollinator_abundance_results[farm_season][fid]['sum'] /
                    float(pollinator_abundance_results[farm_season][fid]['count']))
    
            target_farm_layer.SetFeature(farm_feature)
        target_farm_layer.SyncToDisk()
        target_farm_layer = None
        target_farm_vector = None
    
        task_graph.close()
        task_graph.join()
        
        self.crop_pollination_results_info={'args':args,'paths_dict':paths_dict}
    
    
    def _rasterize_vector_onto_base(self,
            base_raster_path, base_vector_path, attribute_id,
            target_raster_path, filter_string=None):
        """Rasterize attribute from vector onto a copy of base.
    
        Args:
            base_raster_path (string): path to a base raster file
            attribute_id (string): id in `base_vector_path` to rasterize.
            target_raster_path (string): a copy of `base_raster_path` with
                `base_vector_path[attribute_id]` rasterized on top.
            filter_string (string): filtering string to select from farm layer
    
        Returns:
            None.
        """
        base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
        raster_driver = gdal.GetDriverByName('GTiff')
        target_raster = raster_driver.CreateCopy(target_raster_path, base_raster)
        base_raster = None
    
        vector = gdal.OpenEx(base_vector_path)
        layer = vector.GetLayer()
    
        if filter_string is not None:
            layer.SetAttributeFilter(str(filter_string))
        gdal.RasterizeLayer(
            target_raster, [1], layer,
            options=[f'ATTRIBUTE={attribute_id}'])
        target_raster.FlushCache()
        target_raster = None
        layer = None
        vector = None
    
    
    def _create_farm_result_vector(self,
            base_vector_path, target_vector_path):
        """Create a copy of `base_vector_path` and add FID field to it.
    
        Args:
            base_vector_path (string): path to vector to copy
            target_vector_path (string): path to target vector. This path must
                not already exist. Vector will be created at this path that is
                a copy of the base vector with result fields added:
                    pollination._POLLINATOR_ABUNDANCE_FARM_FIELD_ID,
                    pollination._TOTAL_FARM_YIELD_FIELD_ID,
                    pollination._POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID,
                    pollination._WILD_POLLINATOR_FARM_YIELD_FIELD_ID
    
        Returns:
            None.
    
        """
        base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    
        driver = gdal.GetDriverByName('ESRI Shapefile')
        target_vector = driver.CreateCopy(target_vector_path, base_vector)
        target_layer = target_vector.GetLayer()
    
        farm_pollinator_abundance_defn = ogr.FieldDefn(
            _POLLINATOR_ABUNDANCE_FARM_FIELD_ID, ogr.OFTReal)
        farm_pollinator_abundance_defn.SetWidth(25)
        farm_pollinator_abundance_defn.SetPrecision(11)
        target_layer.CreateField(farm_pollinator_abundance_defn)
    
        total_farm_yield_field_defn = ogr.FieldDefn(
            _TOTAL_FARM_YIELD_FIELD_ID, ogr.OFTReal)
        total_farm_yield_field_defn.SetWidth(25)
        total_farm_yield_field_defn.SetPrecision(11)
        target_layer.CreateField(total_farm_yield_field_defn)
    
        pol_proportion_farm_yield_field_defn = ogr.FieldDefn(
            _POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID, ogr.OFTReal)
        pol_proportion_farm_yield_field_defn.SetWidth(25)
        pol_proportion_farm_yield_field_defn.SetPrecision(11)
        target_layer.CreateField(pol_proportion_farm_yield_field_defn)
    
        wild_pol_farm_yield_field_defn = ogr.FieldDefn(
            _WILD_POLLINATOR_FARM_YIELD_FIELD_ID, ogr.OFTReal)
        wild_pol_farm_yield_field_defn.SetWidth(25)
        wild_pol_farm_yield_field_defn.SetPrecision(11)
        target_layer.CreateField(wild_pol_farm_yield_field_defn)
    
        target_layer = None
        target_vector.FlushCache()
        target_vector = None
    
    
    def _parse_scenario_variables(self,args):
        """Parse out scenario variables from input parameters.
    
        This function parses through the guild table, biophysical table, and
        farm polygons (if available) to generate
    
        Parameter:
            args (dict): this is the args dictionary passed in to the `execute`
                function, requires a 'guild_table_path', and
                'landcover_biophysical_table_path' key and optional
                'farm_vector_path' key.
    
        Returns:
            A dictionary with the keys:
                * season_list (list of string)
                * substrate_list (list of string)
                * species_list (list of string)
                * alpha_value[species] (float)
                * landcover_substrate_index[substrate][landcover] (float)
                * landcover_floral_resources[season][landcover] (float)
                * species_abundance[species] (string->float)
                * species_foraging_activity[(species, season)] (string->float)
                * species_substrate_index[(species, substrate)] (tuple->float)
                * foraging_activity_index[(species, season)] (tuple->float)
        """
        guild_table_path = args['guild_table_path']
        landcover_biophysical_table_path = args['landcover_biophysical_table_path']
        if 'farm_vector_path' in args and args['farm_vector_path'] != '':
            farm_vector_path = args['farm_vector_path']
        else:
            farm_vector_path = None
    
        guild_table = utils.build_lookup_from_csv(
            guild_table_path, 'species', to_lower=True)
    
        LOGGER.info('Checking to make sure guild table has all expected headers')
        guild_headers = list(guild_table.values())[0].keys()
        for header in _EXPECTED_GUILD_HEADERS:
            matches = re.findall(header, " ".join(guild_headers))
            if len(matches) == 0:
                raise ValueError(
                    "Expected a header in guild table that matched the pattern "
                    f"'{header}' but was unable to find one. Here are all the "
                    f"headers from {guild_table_path}: {', '.join(guild_headers)}")
    
        landcover_biophysical_table = utils.build_lookup_from_csv(
            landcover_biophysical_table_path, 'lucode', to_lower=True)
        biophysical_table_headers = (
            list(landcover_biophysical_table.values())[0].keys())
        for header in _EXPECTED_BIOPHYSICAL_HEADERS:
            matches = re.findall(header, " ".join(biophysical_table_headers))
            if len(matches) == 0:
                raise ValueError(
                    "Expected a header in biophysical table that matched the "
                    f"pattern '{header}' but was unable to find one. Here are all "
                    f"the headers from {landcover_biophysical_table_path}: "
                    f"{', '.join(biophysical_table_headers)}")
    
        # this dict to dict will map seasons to guild/biophysical headers
        # ex season_to_header['spring']['guilds']
        season_to_header = collections.defaultdict(dict)
        # this dict to dict will map substrate types to guild/biophysical headers
        # ex substrate_to_header['cavity']['biophysical']
        substrate_to_header = collections.defaultdict(dict)
        for header in guild_headers:
            match = re.match(_FORAGING_ACTIVITY_RE_PATTERN, header)
            if match:
                season = match.group(1)
                season_to_header[season]['guild'] = match.group()
            match = re.match(_NESTING_SUITABILITY_PATTERN, header)
            if match:
                substrate = match.group(1)
                substrate_to_header[substrate]['guild'] = match.group()
    
        farm_vector = None
        if farm_vector_path is not None:
            LOGGER.info('Checking that farm polygon has expected headers')
            farm_vector = gdal.OpenEx(farm_vector_path)
            farm_layer = farm_vector.GetLayer()
            if farm_layer.GetGeomType() not in [
                    ogr.wkbPolygon, ogr.wkbMultiPolygon]:
                farm_layer = None
                farm_vector = None
                raise ValueError("Farm layer not a polygon type")
            farm_layer_defn = farm_layer.GetLayerDefn()
            farm_headers = [
                farm_layer_defn.GetFieldDefn(i).GetName()
                for i in range(farm_layer_defn.GetFieldCount())]
            for header in _EXPECTED_FARM_HEADERS:
                matches = re.findall(header, " ".join(farm_headers))
                if not matches:
                    raise ValueError(
                        f"Missing expected header(s) '{header}' from "
                        f"{farm_vector_path}.\n"
                        f"Got these headers instead: {farm_headers}")
    
            for header in farm_headers:
                match = re.match(_FARM_FLORAL_RESOURCES_PATTERN, header)
                if match:
                    season = match.group(1)
                    season_to_header[season]['farm'] = match.group()
                match = re.match(_FARM_NESTING_SUBSTRATE_RE_PATTERN, header)
                if match:
                    substrate = match.group(1)
                    substrate_to_header[substrate]['farm'] = match.group()
    
        for header in biophysical_table_headers:
            match = re.match(_FLORAL_RESOURCES_AVAILABLE_PATTERN, header)
            if match:
                season = match.group(1)
                season_to_header[season]['biophysical'] = match.group()
            match = re.match(_NESTING_SUBSTRATE_PATTERN, header)
            if match:
                substrate = match.group(1)
                substrate_to_header[substrate]['biophysical'] = match.group()
    
        for table_type, lookup_table in itertools.chain(
                season_to_header.items(), substrate_to_header.items()):
            if len(lookup_table) != 3 and farm_vector is not None:
                raise ValueError(
                    "Expected a biophysical, guild, and farm entry for "
                    f"'{table_type}' but instead found only "
                    f"{list(lookup_table.keys())}. Ensure there are "
                    f"corresponding entries of '{table_type}' in both the "
                    "guilds, biophysical table, and farm fields.")
            elif len(lookup_table) != 2 and farm_vector is None:
                raise ValueError(
                    f"Expected a biophysical and guild entry for '{table_type}' "
                    f"but instead found only {list(lookup_table.keys())}. "
                    "Ensure there are corresponding entries of '{table_type}' in "
                    "both the guilds and biophysical table.")
    
        if farm_vector_path is not None:
            farm_season_set = set()
            for farm_feature in farm_layer:
                farm_season_set.add(farm_feature.GetField(_FARM_SEASON_FIELD))
    
            if len(farm_season_set.difference(season_to_header)) > 0:
                raise ValueError(
                    "Found seasons in farm polygon that were not specified in the "
                    "biophysical table: "
                    f"{', '.join(farm_season_set.difference(season_to_header))}. "
                    f"Expected only these: {', '.join(season_to_header.keys())}")
    
        result = {}
        # * season_list (list of string)
        result['season_list'] = sorted(season_to_header)
        # * substrate_list (list of string)
        result['substrate_list'] = sorted(substrate_to_header)
        # * species_list (list of string)
        result['species_list'] = sorted(guild_table)
    
        result['alpha_value'] = dict()
        for species in result['species_list']:
            result['alpha_value'][species] = float(
                guild_table[species][_ALPHA_HEADER])
    
        # * species_abundance[species] (string->float)
        total_relative_abundance = numpy.sum([
            guild_table[species][_RELATIVE_SPECIES_ABUNDANCE_FIELD]
            for species in result['species_list']])
        result['species_abundance'] = {}
        for species in result['species_list']:
            result['species_abundance'][species] = (
                guild_table[species][_RELATIVE_SPECIES_ABUNDANCE_FIELD] /
                float(total_relative_abundance))
    
        # map the relative foraging activity of a species during a certain season
        # (species, season)
        result['species_foraging_activity'] = dict()
        for species in result['species_list']:
            total_activity = numpy.sum([
                guild_table[species][_FORAGING_ACTIVITY_PATTERN % season]
                for season in result['season_list']])
            for season in result['season_list']:
                result['species_foraging_activity'][(species, season)] = (
                    guild_table[species][_FORAGING_ACTIVITY_PATTERN % season] /
                    float(total_activity))
    
        # * landcover_substrate_index[substrate][landcover] (float)
        result['landcover_substrate_index'] = collections.defaultdict(dict)
        for raw_landcover_id in landcover_biophysical_table:
            landcover_id = int(raw_landcover_id)
            for substrate in result['substrate_list']:
                substrate_biophysical_header = (
                    substrate_to_header[substrate]['biophysical'])
                result['landcover_substrate_index'][substrate][landcover_id] = (
                    landcover_biophysical_table[landcover_id][
                        substrate_biophysical_header])
    
        # * landcover_floral_resources[season][landcover] (float)
        result['landcover_floral_resources'] = collections.defaultdict(dict)
        for raw_landcover_id in landcover_biophysical_table:
            landcover_id = int(raw_landcover_id)
            for season in result['season_list']:
                floral_rources_header = season_to_header[season]['biophysical']
                result['landcover_floral_resources'][season][landcover_id] = (
                    landcover_biophysical_table[landcover_id][
                        floral_rources_header])
    
        # * species_substrate_index[(species, substrate)] (tuple->float)
        result['species_substrate_index'] = collections.defaultdict(dict)
        for species in result['species_list']:
            for substrate in result['substrate_list']:
                substrate_guild_header = substrate_to_header[substrate]['guild']
                result['species_substrate_index'][species][substrate] = (
                    guild_table[species][substrate_guild_header])
    
        # * foraging_activity_index[(species, season)] (tuple->float)
        result['foraging_activity_index'] = {}
        for species in result['species_list']:
            for season in result['season_list']:
                key = (species, season)
                foraging_biophyiscal_header = season_to_header[season]['guild']
                result['foraging_activity_index'][key] = (
                    guild_table[species][foraging_biophyiscal_header])
    
        return result


class _CalculateHabitatNestingIndex(object):
    """Closure for HN(x, s) = max_n(N(x, n) ns(s,n)) calculation."""

    def __init__(
            self, substrate_path_map, species_substrate_index_map,
            target_habitat_nesting_index_path):
        """Define parameters necessary for HN(x,s) calculation.

        Args:
            substrate_path_map (dict): map substrate name to substrate index
                raster path. (N(x, n))
            species_substrate_index_map (dict): map substrate name to
                scalar value of species substrate suitability. (ns(s,n))
            target_habitat_nesting_index_path (string): path to target
                raster
        """
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(inspect.getsource(
                    _CalculateHabitatNestingIndex.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = _CalculateHabitatNestingIndex.__name__
        self.__name__ += str([
            substrate_path_map, species_substrate_index_map,
            target_habitat_nesting_index_path])
        self.substrate_path_list = [
            substrate_path_map[substrate_id]
            for substrate_id in sorted(substrate_path_map)]

        self.species_substrate_suitability_index_array = numpy.array([
            species_substrate_index_map[substrate_id]
            for substrate_id in sorted(substrate_path_map)]).reshape(
                (len(species_substrate_index_map), 1))

        self.target_habitat_nesting_index_path = (
            target_habitat_nesting_index_path)

    def __call__(self):
        """Calculate HN(x, s) = max_n(N(x, n) ns(s,n))."""
        def max_op(*substrate_index_arrays):
            """Return the max of index_array[n] * ns[n]."""
            result = numpy.max(
                numpy.stack([x.flatten() for x in substrate_index_arrays]) *
                self.species_substrate_suitability_index_array, axis=0)
            result = result.reshape(substrate_index_arrays[0].shape)
            nodata_mask = utils.array_equals_nodata(
                substrate_index_arrays[0], _INDEX_NODATA)
            result[nodata_mask] = _INDEX_NODATA
            return result

        pygeoprocessing.raster_calculator(
            [(x, 1) for x in self.substrate_path_list], max_op,
            self.target_habitat_nesting_index_path,
            gdal.GDT_Float32, _INDEX_NODATA)


class _SumRasters(object):
    """Sum all rasters where nodata is 0 unless the entire stack is nodata."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _SumRasters.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _SumRasters.__name__)

    def __call__(self, *array_list):
        """Calculate sum of array_list and account for nodata."""
        valid_mask = numpy.zeros(array_list[0].shape, dtype=bool)
        result = numpy.empty_like(array_list[0])
        result[:] = 0
        for array in array_list:
            local_valid_mask = ~utils.array_equals_nodata(array, _INDEX_NODATA)
            result[local_valid_mask] += array[local_valid_mask]
            valid_mask |= local_valid_mask
        result[~valid_mask] = _INDEX_NODATA
        return result


class _PollinatorSupplyOp(object):
    """Calc PA=RA*fa/FR * convolve(PS)."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _PollinatorSupplyOp.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _PollinatorSupplyOp.__name__)

    def __call__(
            self, foraged_flowers_array, floral_resources_array,
            convolve_ps_array):
        """Calculating (RA*fa)/FR * convolve(PS)."""
        valid_mask = ~utils.array_equals_nodata(
            foraged_flowers_array, _INDEX_NODATA)
        result = numpy.empty_like(foraged_flowers_array)
        result[:] = _INDEX_NODATA
        zero_mask = floral_resources_array == 0
        result[zero_mask & valid_mask] = 0
        result_mask = valid_mask & ~zero_mask
        result[result_mask] = (
            foraged_flowers_array[result_mask] /
            floral_resources_array[result_mask] *
            convolve_ps_array[result_mask])
        return result


class _PollinatorSupplyIndexOp(object):
    """Calculate PS(x,s) = FR(x,s) * HN(x,s) * sa(s)."""

    def __init__(self, species_abundance):
        """Create a closure for species abundance to multiply later.

        Args:
            species_abundance (float): value to use in `__call__` when
                calculating pollinator abundance.

        Returns:
            None.
        """
        self.species_abundance = species_abundance
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _PollinatorSupplyIndexOp.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _PollinatorSupplyIndexOp.__name__)
        self.__name__ += str(species_abundance)

    def __call__(
            self, floral_resources_array, habitat_nesting_suitability_array):
        """Calculate f_r * h_n * self.species_abundance."""
        result = numpy.empty_like(floral_resources_array)
        result[:] = _INDEX_NODATA
        valid_mask = ~utils.array_equals_nodata(
            floral_resources_array, _INDEX_NODATA)
        result[valid_mask] = (
            self.species_abundance * floral_resources_array[valid_mask] *
            habitat_nesting_suitability_array[valid_mask])
        return result


class _MultByScalar(object):
    """Calculate a raster * scalar.  Mask through nodata."""

    def __init__(self, scalar):
        """Create a closure for multiplying an array by a scalar.

        Args:
            scalar (float): value to use in `__call__` when multiplying by
                its parameter.

        Returns:
            None.
        """
        self.scalar = scalar
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _MultByScalar.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _MultByScalar.__name__)
        self.__name__ += str(scalar)

    def __call__(self, array):
        """Return array * self.scalar accounting for nodata."""
        result = numpy.empty_like(array)
        result[:] = _INDEX_NODATA
        valid_mask = ~utils.array_equals_nodata(array, _INDEX_NODATA)
        result[valid_mask] = array[valid_mask] * self.scalar
        return result


class _OnFarmPollinatorAbundance(object):
    """Calculate FP(x) = (PAT * (1 - h)) / (h * (1 - 2*pat)+pat))."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _OnFarmPollinatorAbundance.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _OnFarmPollinatorAbundance.__name__)

    def __call__(self, h_array, pat_array):
        """Return (pad * (1 - h)) / (h * (1 - 2*pat)+pat)) tolerate nodata."""
        result = numpy.empty_like(h_array)
        result[:] = _INDEX_NODATA

        valid_mask = (
            ~utils.array_equals_nodata(h_array, _INDEX_NODATA) &
            ~utils.array_equals_nodata(pat_array, _INDEX_NODATA))

        result[valid_mask] = (
            (pat_array[valid_mask]*(1-h_array[valid_mask])) /
            (h_array[valid_mask]*(1-2*pat_array[valid_mask]) +
             pat_array[valid_mask]))
        return result


class _PYTOp(object):
    """Calculate PYT=min((mp+FP), 1)."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _PYTOp.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _PYTOp.__name__)

    def __call__(self, mp_array, FP_array):
        """Return min(mp_array+FP_array, 1) accounting for nodata."""
        valid_mask = ~utils.array_equals_nodata(mp_array, _INDEX_NODATA)
        result = numpy.empty_like(mp_array)
        result[:] = _INDEX_NODATA
        result[valid_mask] = mp_array[valid_mask]+FP_array[valid_mask]
        min_mask = valid_mask & (result > 1)
        result[min_mask] = 1
        return result


class _PYWOp(object):
    """Calculate PYW=max(0,PYT-mp)."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(
                    _PYWOp.__call__
                ).encode('utf-8')).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = (
                _PYWOp.__name__)

    def __call__(self, mp_array, PYT_array):
        """Return max(0,PYT_array-mp_array) accounting for nodata."""
        valid_mask = ~utils.array_equals_nodata(mp_array, _INDEX_NODATA)
        result = numpy.empty_like(mp_array)
        result[:] = _INDEX_NODATA
        result[valid_mask] = PYT_array[valid_mask]-mp_array[valid_mask]
        max_mask = valid_mask & (result < 0)
        result[max_mask] = 0
        return result

    
    
if __name__=="__main__":
    args={
        'workspace_dir':r'I:\ESVs\pollination',
        'results_suffix':'usda',
        'n_workers':6,    
        'landcover_raster_path':r'I:\data\invest\pollination\landcover.tif',
        'guild_table_path':r'I:\data\invest\pollination\guild_table.csv',
        'landcover_biophysical_table_path':r'I:\data\invest\pollination\landcover_biophysical_table.csv',
        'farm_vector_path':r'I:\data\invest\pollination\farms.shp',
        }
    
    cp=Crop_pollination(args)
        
        

