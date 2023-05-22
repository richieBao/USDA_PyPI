# -*- coding: utf-8 -*-
"""InVEST Crop Production Percentile Model."""
import collections
import logging
import os
import re

import numpy
from osgeo import gdal
from osgeo import osr
import pygeoprocessing
import taskgraph

import sys,os
sys.path.append('..')

if __package__:
    from . import _utils as utils
    from ..configs import cfg_crop_production_percentile
else:
    import _utils as utils
    from configs import cfg_crop_production_percentile
    
LOGGER = logging.getLogger(__name__)


_INTERMEDIATE_OUTPUT_DIR = cfg_crop_production_percentile._INTERMEDIATE_OUTPUT_DIR
_YIELD_PERCENTILE_FIELD_PATTERN = cfg_crop_production_percentile._YIELD_PERCENTILE_FIELD_PATTERN
_GLOBAL_OBSERVED_YIELD_FILE_PATTERN = cfg_crop_production_percentile._GLOBAL_OBSERVED_YIELD_FILE_PATTERN
_EXTENDED_CLIMATE_BIN_FILE_PATTERN = cfg_crop_production_percentile._EXTENDED_CLIMATE_BIN_FILE_PATTERN
_CLIMATE_PERCENTILE_TABLE_PATTERN = cfg_crop_production_percentile._CLIMATE_PERCENTILE_TABLE_PATTERN
_INTERPOLATED_YIELD_PERCENTILE_FILE_PATTERN = cfg_crop_production_percentile._INTERPOLATED_YIELD_PERCENTILE_FILE_PATTERN
_CLIPPED_CLIMATE_BIN_FILE_PATTERN = cfg_crop_production_percentile._CLIPPED_CLIMATE_BIN_FILE_PATTERN
_COARSE_YIELD_PERCENTILE_FILE_PATTERN = cfg_crop_production_percentile._COARSE_YIELD_PERCENTILE_FILE_PATTERN
_PERCENTILE_CROP_PRODUCTION_FILE_PATTERN = cfg_crop_production_percentile._PERCENTILE_CROP_PRODUCTION_FILE_PATTERN
_CLIPPED_OBSERVED_YIELD_FILE_PATTERN = cfg_crop_production_percentile._CLIPPED_OBSERVED_YIELD_FILE_PATTERN
_ZEROED_OBSERVED_YIELD_FILE_PATTERN = cfg_crop_production_percentile._ZEROED_OBSERVED_YIELD_FILE_PATTERN
_INTERPOLATED_OBSERVED_YIELD_FILE_PATTERN = cfg_crop_production_percentile._INTERPOLATED_OBSERVED_YIELD_FILE_PATTERN
_OBSERVED_PRODUCTION_FILE_PATTERN = cfg_crop_production_percentile._OBSERVED_PRODUCTION_FILE_PATTERN
_AGGREGATE_VECTOR_FILE_PATTERN = cfg_crop_production_percentile._AGGREGATE_VECTOR_FILE_PATTERN
_AGGREGATE_TABLE_FILE_PATTERN = cfg_crop_production_percentile._AGGREGATE_TABLE_FILE_PATTERN
_EXPECTED_NUTRIENT_TABLE_HEADERS = cfg_crop_production_percentile._EXPECTED_NUTRIENT_TABLE_HEADERS
_EXPECTED_LUCODE_TABLE_HEADER = cfg_crop_production_percentile._EXPECTED_LUCODE_TABLE_HEADER
_NODATA_YIELD = cfg_crop_production_percentile._NODATA_YIELD


class Crop_production_percentile:
    """Crop Production Percentile.

    This model will take a landcover (crop cover?) map and produce yields,
    production, and observed crop yields, a nutrient table, and a clipped
    observed map.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['landcover_raster_path'] (string): path to landcover raster
        args['landcover_to_crop_table_path'] (string): path to a table that
            converts landcover types to crop names that has two headers:

            * lucode: integer value corresponding to a landcover code in
              `args['landcover_raster_path']`.
            * crop_name: a string that must match one of the crops in
              args['model_data_path']/climate_bin_maps/[cropname]_*
              A ValueError is raised if strings don't match.

        args['aggregate_polygon_path'] (string): path to polygon shapefile
            that will be used to aggregate crop yields and total nutrient
            value. (optional, if value is None, then skipped)
        args['model_data_path'] (string): path to the InVEST Crop Production
            global data directory.  This model expects that the following
            directories are subdirectories of this path:

            * climate_bin_maps (contains [cropname]_climate_bin.tif files)
            * climate_percentile_yield (contains
              [cropname]_percentile_yield_table.csv files)

            Please see the InVEST user's guide chapter on crop production for
            details about how to download these data.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None.

    """    
    def __init__(self,args):
        crop_to_landcover_table = utils.build_lookup_from_csv(
            args['landcover_to_crop_table_path'], 'crop_name', to_lower=True)
        bad_crop_name_list = []
        for crop_name in crop_to_landcover_table:
            crop_climate_bin_raster_path = os.path.join(
                args['model_data_path'],
                _EXTENDED_CLIMATE_BIN_FILE_PATTERN % crop_name)
            if not os.path.exists(crop_climate_bin_raster_path):
                bad_crop_name_list.append(crop_name)
        if bad_crop_name_list:
            raise ValueError(
                "The following crop names were provided in %s but no such crops "
                "exist for this model: %s" % (
                    args['landcover_to_crop_table_path'], bad_crop_name_list))
    
        file_suffix = utils.make_suffix_string(args, 'results_suffix')
        output_dir = os.path.join(args['workspace_dir'])
        utils.make_directories([
            output_dir, os.path.join(output_dir, _INTERMEDIATE_OUTPUT_DIR)])
    
        landcover_raster_info = pygeoprocessing.get_raster_info(
            args['landcover_raster_path'])
        pixel_area_ha = numpy.product([
            abs(x) for x in landcover_raster_info['pixel_size']]) / 10000
        landcover_nodata = landcover_raster_info['nodata'][0]
        if landcover_nodata is None:
            LOGGER.warning(
                "%s does not have nodata value defined; "
                "assuming all pixel values are valid"
                % args['landcover_raster_path'])
    
        # Calculate lat/lng bounding box for landcover map
        wgs84srs = osr.SpatialReference()
        wgs84srs.ImportFromEPSG(4326)  # EPSG4326 is WGS84 lat/lng
        landcover_wgs84_bounding_box = pygeoprocessing.transform_bounding_box(
            landcover_raster_info['bounding_box'],
            landcover_raster_info['projection_wkt'], wgs84srs.ExportToWkt(),
            edge_samples=11)
    
        # Initialize a TaskGraph
        work_token_dir = os.path.join(
            output_dir, _INTERMEDIATE_OUTPUT_DIR, '_taskgraph_working_dir')
        try:
            n_workers = int(args['n_workers'])
        except (KeyError, ValueError, TypeError):
            # KeyError when n_workers is not present in args
            # ValueError when n_workers is an empty string.
            # TypeError when n_workers is None.
            n_workers = -1  # Single process mode.
        task_graph = taskgraph.TaskGraph(work_token_dir, n_workers)
        dependent_task_list = []
    
        crop_lucode = None
        observed_yield_nodata = None
        for crop_name in crop_to_landcover_table:
            crop_lucode = crop_to_landcover_table[crop_name][
                _EXPECTED_LUCODE_TABLE_HEADER]
            LOGGER.info("Processing crop %s", crop_name)
            crop_climate_bin_raster_path = os.path.join(
                args['model_data_path'],
                _EXTENDED_CLIMATE_BIN_FILE_PATTERN % crop_name)
    
            LOGGER.info(
                "Clipping global climate bin raster to landcover bounding box.")
            clipped_climate_bin_raster_path = os.path.join(
                output_dir, _CLIPPED_CLIMATE_BIN_FILE_PATTERN % (
                    crop_name, file_suffix))
            crop_climate_bin_raster_info = pygeoprocessing.get_raster_info(
                crop_climate_bin_raster_path)
            crop_climate_bin_task = task_graph.add_task(
                func=pygeoprocessing.warp_raster,
                args=(crop_climate_bin_raster_path,
                      crop_climate_bin_raster_info['pixel_size'],
                      clipped_climate_bin_raster_path, 'near'),
                kwargs={'target_bb': landcover_wgs84_bounding_box},
                target_path_list=[clipped_climate_bin_raster_path],
                task_name='crop_climate_bin')
            dependent_task_list.append(crop_climate_bin_task)
    
            climate_percentile_yield_table_path = os.path.join(
                args['model_data_path'],
                _CLIMATE_PERCENTILE_TABLE_PATTERN % crop_name)
            crop_climate_percentile_table = utils.build_lookup_from_csv(
                climate_percentile_yield_table_path, 'climate_bin', to_lower=True)
            yield_percentile_headers = [
                x for x in list(crop_climate_percentile_table.values())[0]
                if x != 'climate_bin']
    
            reclassify_error_details = {
                'raster_name': f'{crop_name} Climate Bin',
                'column_name': 'climate_bin',
                'table_name': f'Climate {crop_name} Percentile Yield'}
            for yield_percentile_id in yield_percentile_headers:
                LOGGER.info("Map %s to climate bins.", yield_percentile_id)
                interpolated_yield_percentile_raster_path = os.path.join(
                    output_dir,
                    _INTERPOLATED_YIELD_PERCENTILE_FILE_PATTERN % (
                        crop_name, yield_percentile_id, file_suffix))
                bin_to_percentile_yield = dict([
                    (bin_id,
                     crop_climate_percentile_table[bin_id][yield_percentile_id])
                    for bin_id in crop_climate_percentile_table])
                # reclassify nodata to a valid value of 0
                # we're assuming that the crop doesn't exist where there is no data
                # this is more likely than assuming the crop does exist, esp.
                # in the context of the provided climate bins map
                bin_to_percentile_yield[
                    crop_climate_bin_raster_info['nodata'][0]] = 0
                coarse_yield_percentile_raster_path = os.path.join(
                    output_dir,
                    _COARSE_YIELD_PERCENTILE_FILE_PATTERN % (
                        crop_name, yield_percentile_id, file_suffix))
                create_coarse_yield_percentile_task = task_graph.add_task(
                    func=utils.reclassify_raster,
                    args=((clipped_climate_bin_raster_path, 1),
                          bin_to_percentile_yield,
                          coarse_yield_percentile_raster_path, gdal.GDT_Float32,
                          _NODATA_YIELD, reclassify_error_details),
                    target_path_list=[coarse_yield_percentile_raster_path],
                    dependent_task_list=[crop_climate_bin_task],
                    task_name='create_coarse_yield_percentile_%s_%s' % (
                        crop_name, yield_percentile_id))
                dependent_task_list.append(create_coarse_yield_percentile_task)
    
                LOGGER.info(
                    "Interpolate %s %s yield raster to landcover resolution.",
                    crop_name, yield_percentile_id)
                create_interpolated_yield_percentile_task = task_graph.add_task(
                    func=pygeoprocessing.warp_raster,
                    args=(coarse_yield_percentile_raster_path,
                          landcover_raster_info['pixel_size'],
                          interpolated_yield_percentile_raster_path, 'cubicspline'),
                    kwargs={'target_projection_wkt': landcover_raster_info['projection_wkt'],
                            'target_bb': landcover_raster_info['bounding_box']},
                    target_path_list=[interpolated_yield_percentile_raster_path],
                    dependent_task_list=[create_coarse_yield_percentile_task],
                    task_name='create_interpolated_yield_percentile_%s_%s' % (
                        crop_name, yield_percentile_id))
                dependent_task_list.append(
                    create_interpolated_yield_percentile_task)
    
                LOGGER.info(
                    "Calculate yield for %s at %s", crop_name,
                    yield_percentile_id)
                percentile_crop_production_raster_path = os.path.join(
                    output_dir,
                    _PERCENTILE_CROP_PRODUCTION_FILE_PATTERN % (
                        crop_name, yield_percentile_id, file_suffix))
    
                create_percentile_production_task = task_graph.add_task(
                    func=self.calculate_crop_production,
                    args=(
                        args['landcover_raster_path'],
                        interpolated_yield_percentile_raster_path,
                        crop_lucode,
                        pixel_area_ha,
                        percentile_crop_production_raster_path),
                    target_path_list=[percentile_crop_production_raster_path],
                    dependent_task_list=[
                        create_interpolated_yield_percentile_task],
                    task_name='create_percentile_production_%s_%s' % (
                        crop_name, yield_percentile_id))
                dependent_task_list.append(create_percentile_production_task)
    
            LOGGER.info("Calculate observed yield for %s", crop_name)
            global_observed_yield_raster_path = os.path.join(
                args['model_data_path'],
                _GLOBAL_OBSERVED_YIELD_FILE_PATTERN % crop_name)
            global_observed_yield_raster_info = (
                pygeoprocessing.get_raster_info(
                    global_observed_yield_raster_path))
    
            clipped_observed_yield_raster_path = os.path.join(
                output_dir, _CLIPPED_OBSERVED_YIELD_FILE_PATTERN % (
                    crop_name, file_suffix))
            clip_global_observed_yield_task = task_graph.add_task(
                func=pygeoprocessing.warp_raster,
                args=(global_observed_yield_raster_path,
                      global_observed_yield_raster_info['pixel_size'],
                      clipped_observed_yield_raster_path, 'near'),
                kwargs={'target_bb': landcover_wgs84_bounding_box},
                target_path_list=[clipped_observed_yield_raster_path],
                task_name='clip_global_observed_yield_%s_' % crop_name)
            dependent_task_list.append(clip_global_observed_yield_task)
    
            observed_yield_nodata = (
                global_observed_yield_raster_info['nodata'][0])
    
            zeroed_observed_yield_raster_path = os.path.join(
                output_dir, _ZEROED_OBSERVED_YIELD_FILE_PATTERN % (
                    crop_name, file_suffix))
    
            nodata_to_zero_for_observed_yield_task = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=([(clipped_observed_yield_raster_path, 1),
                       (observed_yield_nodata, 'raw')],
                      self._zero_observed_yield_op, zeroed_observed_yield_raster_path,
                      gdal.GDT_Float32, observed_yield_nodata),
                target_path_list=[zeroed_observed_yield_raster_path],
                dependent_task_list=[clip_global_observed_yield_task],
                task_name='nodata_to_zero_for_observed_yield_%s_' % crop_name)
            dependent_task_list.append(nodata_to_zero_for_observed_yield_task)
    
            interpolated_observed_yield_raster_path = os.path.join(
                output_dir, _INTERPOLATED_OBSERVED_YIELD_FILE_PATTERN % (
                    crop_name, file_suffix))
    
            LOGGER.info(
                "Interpolating observed %s raster to landcover.", crop_name)
            interpolate_observed_yield_task = task_graph.add_task(
                func=pygeoprocessing.warp_raster,
                args=(zeroed_observed_yield_raster_path,
                      landcover_raster_info['pixel_size'],
                      interpolated_observed_yield_raster_path, 'cubicspline'),
                kwargs={'target_projection_wkt': landcover_raster_info['projection_wkt'],
                        'target_bb': landcover_raster_info['bounding_box']},
                target_path_list=[interpolated_observed_yield_raster_path],
                dependent_task_list=[nodata_to_zero_for_observed_yield_task],
                task_name='interpolate_observed_yield_to_lulc_%s' % crop_name)
            dependent_task_list.append(interpolate_observed_yield_task)
    
            observed_production_raster_path = os.path.join(
                output_dir, _OBSERVED_PRODUCTION_FILE_PATTERN % (
                    crop_name, file_suffix))
    
            calculate_observed_production_task = task_graph.add_task(
                func=pygeoprocessing.raster_calculator,
                args=([(args['landcover_raster_path'], 1),
                       (interpolated_observed_yield_raster_path, 1),
                       (observed_yield_nodata, 'raw'), (landcover_nodata, 'raw'),
                       (crop_lucode, 'raw'), (pixel_area_ha, 'raw')],
                      self._mask_observed_yield_op, observed_production_raster_path,
                      gdal.GDT_Float32, observed_yield_nodata),
                target_path_list=[observed_production_raster_path],
                dependent_task_list=[interpolate_observed_yield_task],
                task_name='calculate_observed_production_%s' % crop_name)
            dependent_task_list.append(calculate_observed_production_task)
    
        # both 'crop_nutrient.csv' and 'crop' are known data/header values for
        # this model data.
        nutrient_table = utils.build_lookup_from_csv(
            os.path.join(args['model_data_path'], 'crop_nutrient.csv'),
            'crop', to_lower=False)
        result_table_path = os.path.join(
            output_dir, 'result_table%s.csv' % file_suffix)
    
        tabulate_results_task = task_graph.add_task(
            func=self.tabulate_results,
            args=(nutrient_table, yield_percentile_headers,
                  crop_to_landcover_table, pixel_area_ha,
                  args['landcover_raster_path'], landcover_nodata,
                  output_dir, file_suffix, result_table_path),
            target_path_list=[result_table_path],
            dependent_task_list=dependent_task_list,
            task_name='tabulate_results')
    
        if ('aggregate_polygon_path' in args and
                args['aggregate_polygon_path'] not in ['', None]):
            LOGGER.info("aggregating result over query polygon")
            target_aggregate_vector_path = os.path.join(
                output_dir, _AGGREGATE_VECTOR_FILE_PATTERN % (file_suffix))
            aggregate_results_table_path = os.path.join(
                output_dir, _AGGREGATE_TABLE_FILE_PATTERN % file_suffix)
            aggregate_results_task = task_graph.add_task(
                func=self.aggregate_to_polygons,
                args=(args['aggregate_polygon_path'],
                      target_aggregate_vector_path,
                      landcover_raster_info['projection_wkt'],
                      crop_to_landcover_table, nutrient_table,
                      yield_percentile_headers, output_dir, file_suffix,
                      aggregate_results_table_path),
                target_path_list=[target_aggregate_vector_path,
                                  aggregate_results_table_path],
                dependent_task_list=dependent_task_list,
                task_name='aggregate_results_to_polygons')
    
        task_graph.close()
        task_graph.join()
    
    
    def calculate_crop_production(self,lulc_path, yield_path, crop_lucode,
                                  pixel_area_ha, target_path):
        """Calculate crop production for a particular crop.
    
        The resulting production value is:
    
        - nodata, where either the LULC or yield input has nodata
        - 0, where the LULC does not match the given LULC code
        - yield * pixel area, where the given LULC code exists
    
        Args:
            lulc_path (str): path to a raster of LULC codes
            yield_path (str): path of a raster of yields for the crop identified
                by ``crop_lucode``, in units per hectare
            crop_lucode (int): LULC code that identifies the crop of interest in
                the ``lulc_path`` raster.
            pixel_area_ha (number): Pixel area in hectares for both input rasters
            target_path (str): Path to write the output crop production raster
    
        Returns:
            None
        """
    
        lulc_nodata = pygeoprocessing.get_raster_info(lulc_path)['nodata'][0]
        yield_nodata = pygeoprocessing.get_raster_info(yield_path)['nodata'][0]
    
        def _crop_production_op(lulc_array, yield_array):
            """Mask in yields that overlap with `crop_lucode`.
    
            Args:
                lulc_array (numpy.ndarray): landcover raster values
                yield_array (numpy.ndarray): interpolated yield raster values
    
            Returns:
                numpy.ndarray with float values of yields for the current crop
    
            """
            result = numpy.full(lulc_array.shape, _NODATA_YIELD,
                                dtype=numpy.float32)
    
            valid_mask = numpy.full(lulc_array.shape, True)
            if lulc_nodata is not None:
                valid_mask &= ~utils.array_equals_nodata(lulc_array, lulc_nodata)
            if yield_nodata is not None:
                valid_mask &= ~utils.array_equals_nodata(yield_array, yield_nodata)
            result[valid_mask] = 0
    
            lulc_mask = lulc_array == crop_lucode
            result[valid_mask & lulc_mask] = (
                yield_array[valid_mask & lulc_mask] * pixel_area_ha)
            return result
    
        pygeoprocessing.raster_calculator(
            [(lulc_path, 1), (yield_path, 1)],
            _crop_production_op,
            target_path,
            gdal.GDT_Float32, _NODATA_YIELD),
    
    
    def _zero_observed_yield_op(self,observed_yield_array, observed_yield_nodata):
        """Reclassify observed_yield nodata to zero.
    
        Args:
            observed_yield_array (numpy.ndarray): raster values
            observed_yield_nodata (float): raster nodata value
    
        Returns:
            numpy.ndarray with observed yield values
    
        """
        result = numpy.empty(
            observed_yield_array.shape, dtype=numpy.float32)
        result[:] = 0
        valid_mask = slice(None)
        if observed_yield_nodata is not None:
            valid_mask = ~utils.array_equals_nodata(
                observed_yield_array, observed_yield_nodata)
        result[valid_mask] = observed_yield_array[valid_mask]
        return result
    
    
    def _mask_observed_yield_op(self,
            lulc_array, observed_yield_array, observed_yield_nodata,
            landcover_nodata, crop_lucode, pixel_area_ha):
        """Mask total observed yield to crop lulc type.
    
        Args:
            lulc_array (numpy.ndarray): landcover raster values
            observed_yield_array (numpy.ndarray): yield raster values
            observed_yield_nodata (float): yield raster nodata value
            landcover_nodata (float): landcover raster nodata value
            crop_lucode (int): code used to mask in the current crop
            pixel_area_ha (float): area of lulc raster cells (hectares)
    
        Returns:
            numpy.ndarray with float values of yields masked to crop_lucode
    
        """
        result = numpy.empty(lulc_array.shape, dtype=numpy.float32)
        if landcover_nodata is not None:
            result[:] = observed_yield_nodata
            valid_mask = ~utils.array_equals_nodata(lulc_array, landcover_nodata)
            result[valid_mask] = 0
        else:
            result[:] = 0
        lulc_mask = lulc_array == crop_lucode
        result[lulc_mask] = (
            observed_yield_array[lulc_mask] * pixel_area_ha)
        return result
    
    
    def tabulate_results(self,
            nutrient_table, yield_percentile_headers,
            crop_to_landcover_table, pixel_area_ha, landcover_raster_path,
            landcover_nodata, output_dir, file_suffix, target_table_path):
        """Write table with total yield and nutrient results by crop.
    
        This function includes all the operations that write to results_table.csv.
    
        Args:
            nutrient_table (dict): a lookup of nutrient values by crop in the
                form of nutrient_table[<crop>][<nutrient>].
            yield_percentile_headers (list): list of strings indicating percentiles
                at which yield was calculated.
            crop_to_landcover_table (dict): landcover codes keyed by crop names
            pixel_area_ha (float): area of lulc raster cells (hectares)
            landcover_raster_path (string): path to landcover raster
            landcover_nodata (float): landcover raster nodata value
            output_dir (string): the file path to the output workspace.
            file_suffix (string): string to appened to any output filenames.
            target_table_path (string): path to 'result_table.csv' in the output
                workspace
    
        Returns:
            None
    
        """
        LOGGER.info("Generating report table")
        production_percentile_headers = [
            'production_' + re.match(
                _YIELD_PERCENTILE_FIELD_PATTERN,
                yield_percentile_id).group(1) for yield_percentile_id in sorted(
                    yield_percentile_headers)]
        nutrient_headers = [
            nutrient_id + '_' + re.match(
                _YIELD_PERCENTILE_FIELD_PATTERN,
                yield_percentile_id).group(1)
            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS
            for yield_percentile_id in sorted(yield_percentile_headers) + [
                'yield_observed']]
        with open(target_table_path, 'w') as result_table:
            result_table.write(
                'crop,area (ha),' + 'production_observed,' +
                ','.join(production_percentile_headers) + ',' + ','.join(
                    nutrient_headers) + '\n')
            for crop_name in sorted(crop_to_landcover_table):
                result_table.write(crop_name)
                production_lookup = {}
                production_pixel_count = 0
                yield_sum = 0
                observed_production_raster_path = os.path.join(
                    output_dir,
                    _OBSERVED_PRODUCTION_FILE_PATTERN % (
                        crop_name, file_suffix))
    
                LOGGER.info(
                    "Calculating production area and summing observed yield.")
                observed_yield_nodata = pygeoprocessing.get_raster_info(
                    observed_production_raster_path)['nodata'][0]
                for _, yield_block in pygeoprocessing.iterblocks(
                        (observed_production_raster_path, 1)):
    
                    # make a valid mask showing which pixels are not nodata
                    # if nodata value undefined, assume all pixels are valid
                    valid_mask = numpy.full(yield_block.shape, True)
                    if observed_yield_nodata is not None:
                        valid_mask = ~utils.array_equals_nodata(
                            yield_block, observed_yield_nodata)
                    production_pixel_count += numpy.count_nonzero(
                        valid_mask & (yield_block > 0))
                    yield_sum += numpy.sum(yield_block[valid_mask])
                production_area = production_pixel_count * pixel_area_ha
                production_lookup['observed'] = yield_sum
                result_table.write(',%f' % production_area)
                result_table.write(",%f" % yield_sum)
    
                for yield_percentile_id in sorted(yield_percentile_headers):
                    yield_percentile_raster_path = os.path.join(
                        output_dir,
                        _PERCENTILE_CROP_PRODUCTION_FILE_PATTERN % (
                            crop_name, yield_percentile_id, file_suffix))
                    yield_sum = 0
                    for _, yield_block in pygeoprocessing.iterblocks(
                            (yield_percentile_raster_path, 1)):
                        # _NODATA_YIELD will always have a value (defined above)
                        yield_sum += numpy.sum(
                            yield_block[~utils.array_equals_nodata(
                                yield_block, _NODATA_YIELD)])
                    production_lookup[yield_percentile_id] = yield_sum
                    result_table.write(",%f" % yield_sum)
    
                # convert 100g to Mg and fraction left over from refuse
                nutrient_factor = 1e4 * (
                    1 - nutrient_table[crop_name]['Percentrefuse'] / 100)
                for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                    for yield_percentile_id in sorted(yield_percentile_headers):
                        total_nutrient = (
                            nutrient_factor *
                            production_lookup[yield_percentile_id] *
                            nutrient_table[crop_name][nutrient_id])
                        result_table.write(",%f" % (total_nutrient))
                    result_table.write(
                        ",%f" % (
                            nutrient_factor *
                            production_lookup['observed'] *
                            nutrient_table[crop_name][nutrient_id]))
                result_table.write('\n')
    
            total_area = 0
            for _, band_values in pygeoprocessing.iterblocks(
                    (landcover_raster_path, 1)):
                if landcover_nodata is not None:
                    total_area += numpy.count_nonzero(
                        ~utils.array_equals_nodata(band_values, landcover_nodata))
                else:
                    total_area += band_values.size
            result_table.write(
                '\n,total area (both crop and non-crop)\n,%f\n' % (
                    total_area * pixel_area_ha))
    
    
    def aggregate_to_polygons(self,
            base_aggregate_vector_path, target_aggregate_vector_path,
            landcover_raster_projection, crop_to_landcover_table,
            nutrient_table, yield_percentile_headers, output_dir, file_suffix,
            target_aggregate_table_path):
        """Write table with aggregate results of yield and nutrient values.
    
        Use zonal statistics to summarize total observed and interpolated
        production and nutrient information for each polygon in
        base_aggregate_vector_path.
    
        Args:
            base_aggregate_vector_path (string): path to polygon vector
            target_aggregate_vector_path (string):
                path to re-projected copy of polygon vector
            landcover_raster_projection (string): a WKT projection string
            crop_to_landcover_table (dict): landcover codes keyed by crop names
            nutrient_table (dict): a lookup of nutrient values by crop in the
                form of nutrient_table[<crop>][<nutrient>].
            yield_percentile_headers (list): list of strings indicating percentiles
                at which yield was calculated.
            output_dir (string): the file path to the output workspace.
            file_suffix (string): string to appened to any output filenames.
            target_aggregate_table_path (string): path to 'aggregate_results.csv'
                in the output workspace
    
        Returns:
            None
    
        """
        # reproject polygon to LULC's projection
        pygeoprocessing.reproject_vector(
            base_aggregate_vector_path,
            landcover_raster_projection,
            target_aggregate_vector_path,
            driver_name='ESRI Shapefile')
    
        # loop over every crop and query with pgp function
        total_yield_lookup = {}
        total_nutrient_table = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(
                float)))
        for crop_name in crop_to_landcover_table:
            # convert 100g to Mg and fraction left over from refuse
            nutrient_factor = 1e4 * (
                1 - nutrient_table[crop_name]['Percentrefuse'] / 100)
            # loop over percentiles
            for yield_percentile_id in yield_percentile_headers:
                percentile_crop_production_raster_path = os.path.join(
                    output_dir,
                    _PERCENTILE_CROP_PRODUCTION_FILE_PATTERN % (
                        crop_name, yield_percentile_id, file_suffix))
                LOGGER.info(
                    "Calculating zonal stats for %s  %s", crop_name,
                    yield_percentile_id)
                total_yield_lookup['%s_%s' % (
                    crop_name, yield_percentile_id)] = (
                        pygeoprocessing.zonal_statistics(
                            (percentile_crop_production_raster_path, 1),
                            target_aggregate_vector_path))
    
                for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                    for id_index in total_yield_lookup['%s_%s' % (
                            crop_name, yield_percentile_id)]:
                        total_nutrient_table[nutrient_id][
                            yield_percentile_id][id_index] += (
                                nutrient_factor *
                                total_yield_lookup['%s_%s' % (
                                    crop_name, yield_percentile_id)][
                                        id_index]['sum'] *
                                nutrient_table[crop_name][nutrient_id])
    
            # process observed
            observed_yield_path = os.path.join(
                output_dir, _OBSERVED_PRODUCTION_FILE_PATTERN % (
                    crop_name, file_suffix))
            total_yield_lookup['%s_observed' % crop_name] = (
                pygeoprocessing.zonal_statistics(
                    (observed_yield_path, 1),
                    target_aggregate_vector_path))
            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                for id_index in total_yield_lookup['%s_observed' % crop_name]:
                    total_nutrient_table[
                        nutrient_id]['observed'][id_index] += (
                            nutrient_factor *
                            total_yield_lookup[
                                '%s_observed' % crop_name][id_index]['sum'] *
                            nutrient_table[crop_name][nutrient_id])
    
        # report everything to a table
        with open(target_aggregate_table_path, 'w') as aggregate_table:
            # write header
            aggregate_table.write('FID,')
            aggregate_table.write(','.join(sorted(total_yield_lookup)) + ',')
            aggregate_table.write(
                ','.join([
                    '%s_%s' % (nutrient_id, model_type)
                    for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS
                    for model_type in sorted(
                        list(total_nutrient_table.values())[0])]))
            aggregate_table.write('\n')
    
            # iterate by polygon index
            for id_index in list(total_yield_lookup.values())[0]:
                aggregate_table.write('%s,' % id_index)
                aggregate_table.write(','.join([
                    str(total_yield_lookup[yield_header][id_index]['sum'])
                    for yield_header in sorted(total_yield_lookup)]))
    
                for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                    for model_type in sorted(
                            list(total_nutrient_table.values())[0]):
                        aggregate_table.write(
                            ',%s' % total_nutrient_table[
                                nutrient_id][model_type][id_index])
                aggregate_table.write('\n')
    
    
if __name__=="__main__":
    args={
        'workspace_dir':r'I:\ESVs\crop_PP',
        'n_workers':6,
        'results_suffix':'usda',
        'landcover_raster_path':r'I:\data\invest\CropProduction\sample_user_data\landcover.tif',
        'landcover_to_crop_table_path':r'I:\data\invest\CropProduction\sample_user_data\landcover_to_crop_table.csv',
        'aggregate_polygon_path':r'I:\data\invest\CropProduction\sample_user_data\aggregate_shape.shp',
        'model_data_path':r'I:\data\invest\CropProduction\model_data',
        }
    
    cpp=Crop_production_percentile(args)