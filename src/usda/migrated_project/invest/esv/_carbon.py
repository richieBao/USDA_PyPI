# -*- coding: utf-8 -*-
"""
Created on Sun May 21 09:10:12 2023

@author: richie bao

ref:https://github.com/natcap/invest
"Carbon Storage and Sequestration.
"""

"""InVEST specific code utils."""
"""Carbon Storage and Sequestration."""
import codecs
import logging
import os
import time
from functools import reduce

from osgeo import gdal
import numpy
import pygeoprocessing
import taskgraph
import taskgraph

import sys,os
sys.path.append('..')

if __package__:
    from . import _utils as utils
    from ..configs import cfg_carbon
else:
    import _utils as utils
    from configs import cfg_carbon

LOGGER = logging.getLogger(__name__)

# -1.0 since carbon stocks are 0 or greater
_CARBON_NODATA = cfg_carbon._CARBON_NODATA
# use min float32 which is unlikely value to see in a NPV raster
_VALUE_NODATA = cfg_carbon._VALUE_NODATA

_OUTPUT_BASE_FILES=cfg_carbon._OUTPUT_BASE_FILES
_INTERMEDIATE_BASE_FILES=cfg_carbon._INTERMEDIATE_BASE_FILES
_TMP_BASE_FILES=cfg_carbon._TMP_BASE_FILES

class Carbon_storage_sequestration:
    """Carbon.

    Calculate the amount of carbon stocks given a landscape, or the difference
    due to a future change, and/or the tradeoffs between that and a REDD
    scenario, and calculate economic valuation on those scenarios.

    The model can operate on a single scenario, a combined present and future
    scenario, as well as an additional REDD scenario.

    Args:
        args['workspace_dir'] (string): a path to the directory that will
            write output and other temporary files during calculation.
        args['results_suffix'] (string): appended to any output file name.
        args['lulc_cur_path'] (string): a path to a raster representing the
            current carbon stocks.
        args['calc_sequestration'] (bool): if true, sequestration should
            be calculated and 'lulc_fut_path' and 'do_redd' should be defined.
        args['lulc_fut_path'] (string): a path to a raster representing future
            landcover scenario.  Optional, but if present and well defined
            will trigger a sequestration calculation.
        args['do_redd'] ( bool): if true, REDD analysis should be calculated
            and 'lulc_redd_path' should be defined
        args['lulc_redd_path'] (string): a path to a raster representing the
            alternative REDD scenario which is only possible if the
            args['lulc_fut_path'] is present and well defined.
        args['carbon_pools_path'] (string): path to CSV or that indexes carbon
            storage density to lulc codes. (required if 'do_uncertainty' is
            false)
        args['lulc_cur_year'] (int/string): an integer representing the year
            of `args['lulc_cur_path']` used if `args['do_valuation']`
            is True.
        args['lulc_fut_year'](int/string): an integer representing the year
            of `args['lulc_fut_path']` used in valuation if it exists.
            Required if  `args['do_valuation']` is True and
            `args['lulc_fut_path']` is present and well defined.
        args['do_valuation'] (bool): if true then run the valuation model on
            available outputs. Calculate NPV for a future scenario or a REDD
            scenario and report in final HTML document.
        args['price_per_metric_ton_of_c'] (float): Is the present value of
            carbon per metric ton. Used if `args['do_valuation']` is present
            and True.
        args['discount_rate'] (float): Discount rate used if NPV calculations
            are required.  Used if `args['do_valuation']` is  present and
            True.
        args['rate_change'] (float): Annual rate of change in price of carbon
            as a percentage.  Used if `args['do_valuation']` is  present and
            True.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None.
    """  
    
    def __init__(self,args):            
        file_suffix = utils.make_suffix_string(args, 'results_suffix')
        # print('---',file_suffix)
        intermediate_output_dir=os.path.join(args['workspace_dir'], 'intermediate_outputs')
        output_dir = args['workspace_dir']
        utils.make_directories([intermediate_output_dir, output_dir])
        
        LOGGER.info('Building file registry')
        
        file_registry = utils.build_file_registry(
            [(_OUTPUT_BASE_FILES, output_dir),
             (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
             (_TMP_BASE_FILES, output_dir)], file_suffix)    
        
        # print(file_registry)
        carbon_pool_table = utils.build_lookup_from_csv(args['carbon_pools_path'], 'lucode')        
        # print(self.carbon_pool_table)

        work_token_dir = os.path.join(intermediate_output_dir, '_taskgraph_working_dir')        
        try:
            n_workers = int(args['n_workers'])
        except (KeyError, ValueError, TypeError):
            # KeyError when n_workers is not present in args
            # ValueError when n_workers is an empty string.
            # TypeError when n_workers is None.
            n_workers = -1  # Synchronous mode.        
        # print(n_workers) 
        graph = taskgraph.TaskGraph(work_token_dir, n_workers)       

        cell_size_set = set()
        raster_size_set = set()
        valid_lulc_keys = []
        valid_scenarios = []
        tifs_to_summarize = set()  # passed to _generate_report()          
        
        for scenario_type in ['cur', 'fut', 'redd']:
            lulc_key = "lulc_%s_path" % (scenario_type)
            if lulc_key in args and args[lulc_key]:
                raster_info = pygeoprocessing.get_raster_info(args[lulc_key])
                cell_size_set.add(raster_info['pixel_size'])
                raster_size_set.add(raster_info['raster_size'])
                valid_lulc_keys.append(lulc_key)
                valid_scenarios.append(scenario_type)    
        
        print(f'cell size set={cell_size_set};\nraster size set={raster_size_set};\nvalid lulc keys={valid_lulc_keys}\nvalid scenarios={valid_scenarios}')
        if len(cell_size_set) > 1:
            raise ValueError(
                "the pixel sizes of %s are not equivalent. Here are the "
                "different sets that were found in processing: %s" % (
                    valid_lulc_keys, cell_size_set))
        if len(raster_size_set) > 1:
            raise ValueError(
                "the raster dimensions of %s are not equivalent. Here are the "
                "different sizes that were found in processing: %s" % (
                    valid_lulc_keys, raster_size_set))  
            
        # calculate total carbon storage
        LOGGER.info('Map all carbon pools to carbon storage rasters.')
        carbon_map_task_lookup = {}
        sum_rasters_task_lookup = {}
        for scenario_type in valid_scenarios:        
            carbon_map_task_lookup[scenario_type] = []
            storage_path_list = []
            for pool_type in ['c_above', 'c_below', 'c_soil', 'c_dead']:
                carbon_pool_by_type = dict([(lucode, float(carbon_pool_table[lucode][pool_type])) for lucode in carbon_pool_table])                
                
                lulc_key = 'lulc_%s_path' % scenario_type
                storage_key = '%s_%s' % (pool_type, scenario_type)
                
                LOGGER.info("Mapping carbon from '%s' to '%s' scenario.",lulc_key, storage_key) 
                
                carbon_map_task = graph.add_task(
                    self._generate_carbon_map,
                    args=(
                        args[lulc_key], 
                        carbon_pool_by_type,
                        file_registry[storage_key]),
                        target_path_list=[file_registry[storage_key]],
                        task_name='carbon_map_%s' % storage_key)
                
                storage_path_list.append(file_registry[storage_key])
                carbon_map_task_lookup[scenario_type].append(carbon_map_task)   
            
            output_key = 'tot_c_' + scenario_type
            LOGGER.info("Calculate carbon storage for '%s'", output_key)            
                
            sum_rasters_task = graph.add_task(
                self._sum_rasters,
                args=(
                    storage_path_list, 
                    file_registry[output_key]),
                
                    target_path_list=[file_registry[output_key]],
                    dependent_task_list=carbon_map_task_lookup[scenario_type],
                    task_name='sum_rasters_for_total_c_%s' % output_key)
            
            sum_rasters_task_lookup[scenario_type] = sum_rasters_task
            tifs_to_summarize.add(file_registry[output_key])            
            
        # calculate sequestration
        diff_rasters_task_lookup = {}
        for scenario_type in ['fut', 'redd']:
            if scenario_type not in valid_scenarios:
                continue
            output_key = 'delta_cur_' + scenario_type
            LOGGER.info("Calculate sequestration scenario '%s'", output_key)
            storage_path_list = [
                file_registry['tot_c_cur'],
                file_registry['tot_c_' + scenario_type]]
    
            diff_rasters_task = graph.add_task(
                self._diff_rasters,
                args=(
                    storage_path_list, 
                    file_registry[output_key]),
                
                    target_path_list=[file_registry[output_key]],
                    dependent_task_list=[
                    sum_rasters_task_lookup['cur'],
                    sum_rasters_task_lookup[scenario_type]],
                task_name='diff_rasters_for_%s' % output_key)
            
            diff_rasters_task_lookup[scenario_type] = diff_rasters_task
            tifs_to_summarize.add(file_registry[output_key])        
       
        # calculate net present value
        calculate_npv_tasks = []
        if 'do_valuation' in args and args['do_valuation']:
            LOGGER.info('Constructing valuation formula.')
            valuation_constant =self._calculate_valuation_constant(
                int(args['lulc_cur_year']), 
                int(args['lulc_fut_year']),
                float(args['discount_rate']), 
                float(args['rate_change']),
                float(args['price_per_metric_ton_of_c']))    
            
            # print(valuation_constant)
            for scenario_type in ['fut', 'redd']:
                if scenario_type not in valid_scenarios:
                    continue
                output_key = 'npv_%s' % scenario_type
                LOGGER.info("Calculating NPV for scenario '%s'", output_key)             
                
                calculate_npv_task = graph.add_task(
                    self._calculate_npv,
                    args=(file_registry['delta_cur_%s' % scenario_type],
                          valuation_constant, file_registry[output_key]),
                    target_path_list=[file_registry[output_key]],
                    dependent_task_list=[diff_rasters_task_lookup[scenario_type]],
                    task_name='calculate_%s' % output_key)
                
                calculate_npv_tasks.append(calculate_npv_task)
                tifs_to_summarize.add(file_registry[output_key])                
                
        # Report aggregate results
        tasks_to_report = (list(sum_rasters_task_lookup.values())
                           + list(diff_rasters_task_lookup.values())
                           + calculate_npv_tasks)        
                
        _ = graph.add_task(
            self._generate_report,
            args=(tifs_to_summarize, args, file_registry),
            target_path_list=[file_registry['html_report']],
            dependent_task_list=tasks_to_report,
            task_name='generate_report')        
        
        
        graph.close()
        graph.join()
        
        # print(tasks_to_report)
        self.carbon_results_info={'tifs_to_summarize':tifs_to_summarize,
                                  'args':args,
                                  'file_registry':file_registry,}         
        
        
        
    def _generate_carbon_map(self,lulc_path, carbon_pool_by_type, out_carbon_stock_path):
        """Generate carbon stock raster by mapping LULC values to carbon pools.
    
        Args:
            lulc_path (string): landcover raster with integer pixels.
            out_carbon_stock_path (string): path to output raster that will have
                pixels with carbon storage values in them with units of Mg*C
            carbon_pool_by_type (dict): a dictionary that maps landcover values
                to carbon storage densities per area (Mg C/Ha).
    
        Returns:
            None.
        """
        # print('-+'*50)
        lulc_info = pygeoprocessing.get_raster_info(lulc_path)
        pixel_area = abs(numpy.prod(lulc_info['pixel_size']))
        # print(pixel_area)
        carbon_stock_by_type = dict([
            (lulcid, stock * pixel_area / 10**4)
            for lulcid, stock in carbon_pool_by_type.items()])
        # print(carbon_stock_by_type)
        reclass_error_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Carbon Pools'}
        utils.reclassify_raster(
            (lulc_path, 1), carbon_stock_by_type, out_carbon_stock_path,gdal.GDT_Float32, _CARBON_NODATA, reclass_error_details)            
                
    def _sum_rasters(self,storage_path_list, output_sum_path):
        """Sum all the rasters in `storage_path_list` to `output_sum_path`."""
        def _sum_op(*storage_arrays):
            """Sum all the arrays or nodata a pixel stack if one exists."""
            valid_mask = reduce(
                lambda x, y: x & y, [~utils.array_equals_nodata(_, _CARBON_NODATA) for _ in storage_arrays])
            result = numpy.empty(storage_arrays[0].shape)
            result[:] = _CARBON_NODATA
            result[valid_mask] = numpy.sum([_[valid_mask] for _ in storage_arrays], axis=0)
            return result
    
        pygeoprocessing.raster_calculator([(x, 1) for x in storage_path_list], _sum_op, output_sum_path,gdal.GDT_Float32, _CARBON_NODATA)     
       
    def _diff_rasters(self,storage_path_list, output_diff_path):
        """Subtract rasters in `storage_path_list` to `output_sum_path`."""
        def _diff_op(base_array, future_array):
            """Subtract future_array from base_array and ignore nodata."""
            result = numpy.empty(base_array.shape, dtype=numpy.float32)
            result[:] = _CARBON_NODATA
            valid_mask = (
                ~utils.array_equals_nodata(base_array, _CARBON_NODATA) &
                ~utils.array_equals_nodata(future_array, _CARBON_NODATA))
            result[valid_mask] = (
                future_array[valid_mask] - base_array[valid_mask])
            return result
    
        pygeoprocessing.raster_calculator(
            [(x, 1) for x in storage_path_list], _diff_op, output_diff_path,
            gdal.GDT_Float32, _CARBON_NODATA)      
        
    def _calculate_valuation_constant(self,lulc_cur_year, lulc_fut_year, discount_rate, rate_change,price_per_metric_ton_of_c):
        """Calculate a net present valuation constant to multiply carbon storage.
    
        Args:
            lulc_cur_year (int): calendar year in present
            lulc_fut_year (int): calendar year in future
            discount_rate (float): annual discount rate as a percentage
            rate_change (float): annual change in price of carbon as a percentage
            price_per_metric_ton_of_c (float): currency amount of Mg of carbon
    
        Returns:
            a floating point number that can be used to multiply a delta carbon
            storage value by to calculate NPV.
        """
        n_years = lulc_fut_year - lulc_cur_year
        ratio = (
            1 / ((1 + discount_rate / 100) *
                 (1 + rate_change / 100)))
        valuation_constant = (price_per_metric_ton_of_c / n_years)
        # note: the valuation formula in the user's guide uses sum notation.
        # here it's been simplified to remove the sum using the general rule
        # sum(r^k) from k=0 to N  ==  (r^(N+1) - 1) / (r - 1)
        # where N = n_years-1 and r = ratio
        if ratio == 1:
            # if ratio == 1, we would divide by zero in the equation below
            # so use the limit as ratio goes to 1, which is n_years
            valuation_constant *= n_years
        else:
            valuation_constant *= (1 - ratio ** n_years) / (1 - ratio)
        return valuation_constant      
    
    def _calculate_npv(self,delta_carbon_path, valuation_constant, npv_out_path):
        """Calculate net present value.
    
        Args:
            delta_carbon_path (string): path to change in carbon storage over
                time.
            valuation_constant (float): value to multiply each carbon storage
                value by to calculate NPV.
            npv_out_path (string): path to output net present value raster.
    
        Returns:
            None.
        """
        def _npv_value_op(carbon_array):
            """Calculate the NPV given carbon storage or loss values."""
            result = numpy.empty(carbon_array.shape, dtype=numpy.float32)
            result[:] = _VALUE_NODATA
            valid_mask = ~utils.array_equals_nodata(carbon_array,  _CARBON_NODATA)
            result[valid_mask] = carbon_array[valid_mask] * valuation_constant
            return result
    
        pygeoprocessing.raster_calculator(
            [(delta_carbon_path, 1)], _npv_value_op, npv_out_path,
            gdal.GDT_Float32, _VALUE_NODATA)    
        
    
    def _generate_report(self,raster_file_set, model_args, file_registry):
        """Generate a human readable HTML report of summary stats of model run.
    
        Args:
            raster_file_set (set): paths to rasters that need summary stats.
            model_args (dict): InVEST argument dictionary.
            file_registry (dict): file path dictionary for InVEST workspace.
    
        Returns:
            None.
        """
        html_report_path = file_registry['html_report']
        with codecs.open(html_report_path, 'w', encoding='utf-8') as report_doc:
            # Boilerplate header that defines style and intro header.
            header = (
                '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Carbon R'
                'esults</title><style type="text/css">body { background-color: #E'
                'FECCA; color: #002F2F} h1 { text-align: center } h1, h2, h3, h4,'
                'strong, th { color: #046380; } h2 { border-bottom: 1px solid #A7'
                'A37E; } table { border: 5px solid #A7A37E; margin-bottom: 50px; '
                'background-color: #E6E2AF; } td, th { margin-left: 0px; margin-r'
                'ight: 0px; padding-left: 8px; padding-right: 8px; padding-bottom'
                ': 2px; padding-top: 2px; text-align:left; } td { border-top: 5px'
                'solid #EFECCA; } .number {text-align: right; font-family: monosp'
                'ace;} img { margin: 20px; }</style></head><body><h1>InVEST Carbo'
                'n Model Results</h1><p>This document summarizes the results from'
                'running the InVEST carbon model with the following data.</p>')
    
            report_doc.write(header)
            report_doc.write('<p>Report generated at %s</p>' % (
                time.strftime("%Y-%m-%d %H:%M")))
    
            # Report input arguments
            report_doc.write('<table><tr><th>arg id</th><th>arg value</th></tr>')
            for key, value in model_args.items():
                report_doc.write('<tr><td>%s</td><td>%s</td></tr>' % (key, value))
            report_doc.write('</table>')
    
            # Report aggregate results
            report_doc.write('<h3>Aggregate Results</h3>')
            report_doc.write(
                '<table><tr><th>Description</th><th>Value</th><th>Units</th><th>R'
                'aw File</th></tr>')
    
            # value lists are [sort priority, description, statistic, units]
            report = [
                (file_registry['tot_c_cur'], 'Total cur', 'Mg of C'),
                (file_registry['tot_c_fut'], 'Total fut', 'Mg of C'),
                (file_registry['tot_c_redd'], 'Total redd', 'Mg of C'),
                (file_registry['delta_cur_fut'], 'Change in C for fut', 'Mg of C'),
                (file_registry['delta_cur_redd'],
                 'Change in C for redd', 'Mg of C'),
                (file_registry['npv_fut'],
                 'Net present value from cur to fut', 'currency units'),
                (file_registry['npv_redd'],
                 'Net present value from cur to redd', 'currency units'),
            ]
    
            for raster_uri, description, units in report:
                if raster_uri in raster_file_set:
                    summary_stat =self._accumulate_totals(raster_uri)
                    report_doc.write(
                        '<tr><td>%s</td><td class="number">%.2f</td><td>%s</td>'
                        '<td>%s</td></tr>' % (
                            description, summary_stat, units, raster_uri))
            report_doc.write('</body></html>')

    def _accumulate_totals(self,raster_path):
        """Sum all non-nodata pixels in `raster_path` and return result."""
        nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
        raster_sum = 0.0
        for _, block in pygeoprocessing.iterblocks((raster_path, 1)):
            # The float64 dtype in the sum is needed to reduce numerical error in
            # the sum.  Users calculated the sum with ArcGIS zonal statistics,
            # noticed a difference and wrote to us about it on the forum.
            raster_sum += numpy.sum(
                block[~utils.array_equals_nodata(
                        block, nodata)], dtype=numpy.float64)
        return raster_sum       
       
    
if __name__=="__main__":    
    args={
        'workspace_dir':'I:\ESVs\carbon',
        'carbon_pools_path':r'I:\data\invest\Carbon\carbon_pools_willamette.csv',
        'n_workers':6,
        'results_suffix':'usda',
        'lulc_cur_path':r'I:\data\invest\Carbon\lulc_current_willamette.tif',
        'lulc_fut_path':r'I:\data\invest\Carbon\lulc_future_willamette.tif',
        'lulc_redd_path':r"I:\data\invest\Carbon\lulc_redd_willamette.tif",
        'do_valuation':True,
        'lulc_cur_year':2009,
        'lulc_fut_year':2010,
        'discount_rate':0.07,
        'rate_change':0.1,
        'price_per_metric_ton_of_c':100,
        }
    
    
    carbon=Carbon_storage_sequestration(args)
    
