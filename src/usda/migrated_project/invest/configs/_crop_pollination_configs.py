# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:51:58 2023

@author: richie bao
"""

_INDEX_NODATA = -1

# These patterns are expected in the biophysical table
_NESTING_SUBSTRATE_PATTERN = 'nesting_([^_]+)_availability_index'
_FLORAL_RESOURCES_AVAILABLE_PATTERN = 'floral_resources_([^_]+)_index'
_EXPECTED_BIOPHYSICAL_HEADERS = [
    'lucode', _NESTING_SUBSTRATE_PATTERN, _FLORAL_RESOURCES_AVAILABLE_PATTERN]

# These are patterns expected in the guilds table
_NESTING_SUITABILITY_PATTERN = 'nesting_suitability_([^_]+)_index'
# replace with season
_FORAGING_ACTIVITY_PATTERN = 'foraging_activity_%s_index'
_FORAGING_ACTIVITY_RE_PATTERN = _FORAGING_ACTIVITY_PATTERN % '([^_]+)'
_RELATIVE_SPECIES_ABUNDANCE_FIELD = 'relative_abundance'
_ALPHA_HEADER = 'alpha'
_EXPECTED_GUILD_HEADERS = [
    'species', _NESTING_SUITABILITY_PATTERN, _FORAGING_ACTIVITY_RE_PATTERN,
    _ALPHA_HEADER, _RELATIVE_SPECIES_ABUNDANCE_FIELD]

_NESTING_SUBSTRATE_INDEX_FILEPATTERN = 'nesting_substrate_index_%s%s.tif'
# this is used if there is a farm polygon present
_FARM_NESTING_SUBSTRATE_INDEX_FILEPATTERN = (
    'farm_nesting_substrate_index_%s%s.tif')

# replaced by (species, file_suffix)
_HABITAT_NESTING_INDEX_FILE_PATTERN = 'habitat_nesting_index_%s%s.tif'
# replaced by (season, file_suffix)
_RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN = (
    'relative_floral_abundance_index_%s%s.tif')
# this is used if there's a farm polygon present
_FARM_RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN = (
    'farm_relative_floral_abundance_index_%s%s.tif')
# used as an intermediate step for floral resources calculation
# replace (species, file_suffix)
_LOCAL_FORAGING_EFFECTIVENESS_FILE_PATTERN = (
    'local_foraging_effectiveness_%s%s.tif')
# for intermediate output of floral resources replace (species, file_suffix)
_FLORAL_RESOURCES_INDEX_FILE_PATTERN = (
    'floral_resources_%s%s.tif')
# pollinator supply raster replace (species, file_suffix)
_POLLINATOR_SUPPLY_FILE_PATTERN = 'pollinator_supply_%s%s.tif'
# name of reprojected farm vector replace (file_suffix)
_PROJECTED_FARM_VECTOR_FILE_PATTERN = 'reprojected_farm_vector%s.shp'
# used to store the 2D decay kernel for a given distance replace
# (alpha, file suffix)
_KERNEL_FILE_PATTERN = 'kernel_%f%s.tif'
# PA(x,s,j) replace (species, season, file_suffix)
_POLLINATOR_ABUNDANCE_FILE_PATTERN = 'pollinator_abundance_%s_%s%s.tif'
# PAT(x,j) total pollinator abundance per season replace (season, file_suffix)
_TOTAL_POLLINATOR_ABUNDANCE_FILE_PATTERN = (
    'total_pollinator_abundance_%s%s.tif')
# used for RA(l(x),j)*fa(s,j) replace (species, season, file_suffix)
_FORAGED_FLOWERS_INDEX_FILE_PATTERN = (
    'foraged_flowers_index_%s_%s%s.tif')
# used for convolving PS over alpha s replace (species, file_suffix)
_CONVOLVE_PS_FILE_PATH = 'convolve_ps_%s%s.tif'
# half saturation raster replace (season, file_suffix)
_HALF_SATURATION_FILE_PATTERN = 'half_saturation_%s%s.tif'
# blank raster as a basis to rasterize on replace (file_suffix)
_BLANK_RASTER_FILE_PATTERN = 'blank_raster%s.tif'
# raster to hold seasonal farm pollinator replace (season, file_suffix)
_FARM_POLLINATOR_SEASON_FILE_PATTERN = 'farm_pollinator_%s%s.tif'
# total farm pollinators replace (file_suffix)
_FARM_POLLINATOR_FILE_PATTERN = 'farm_pollinators%s.tif'
# managed pollinator indexes replace (file_suffix)
_MANAGED_POLLINATOR_FILE_PATTERN = 'managed_pollinators%s.tif'
# total pollinator raster replace (file_suffix)
_TOTAL_POLLINATOR_YIELD_FILE_PATTERN = 'total_pollinator_yield%s.tif'
# wild pollinator raster replace (file_suffix)
_WILD_POLLINATOR_YIELD_FILE_PATTERN = 'wild_pollinator_yield%s.tif'
# final aggregate farm shapefile file pattern replace (file_suffix)
_FARM_VECTOR_RESULT_FILE_PATTERN = 'farm_results%s.shp'
# output field on target shapefile if farms are enabled
_TOTAL_FARM_YIELD_FIELD_ID = 'y_tot'
# output field for wild pollinators on farms if farms are enabled
_WILD_POLLINATOR_FARM_YIELD_FIELD_ID = 'y_wild'
# output field for proportion of wild pollinators over the pollinator
# dependent part of the yield
_POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID = 'pdep_y_w'
# output field for pollinator abundance on farm for the season of pollination
_POLLINATOR_ABUNDANCE_FARM_FIELD_ID = 'p_abund'
# expected pattern for seasonal floral resources in input shapefile (season)
_FARM_FLORAL_RESOURCES_HEADER_PATTERN = 'fr_%s'
# regular expression version of _FARM_FLORAL_RESOURCES_PATTERN
_FARM_FLORAL_RESOURCES_PATTERN = (
    _FARM_FLORAL_RESOURCES_HEADER_PATTERN % '([^_]+)')
# expected pattern for nesting substrate in input shapfile (substrate)
_FARM_NESTING_SUBSTRATE_HEADER_PATTERN = 'n_%s'
# regular expression version of _FARM_NESTING_SUBSTRATE_HEADER_PATTERN
_FARM_NESTING_SUBSTRATE_RE_PATTERN = (
    _FARM_NESTING_SUBSTRATE_HEADER_PATTERN % '([^_]+)')
_HALF_SATURATION_FARM_HEADER = 'half_sat'
_CROP_POLLINATOR_DEPENDENCE_FIELD = 'p_dep'
_MANAGED_POLLINATORS_FIELD = 'p_managed'
_FARM_SEASON_FIELD = 'season'
_EXPECTED_FARM_HEADERS = [
    _FARM_SEASON_FIELD, 'crop_type', _HALF_SATURATION_FARM_HEADER,
    _MANAGED_POLLINATORS_FIELD, _FARM_FLORAL_RESOURCES_PATTERN,
    _FARM_NESTING_SUBSTRATE_RE_PATTERN, _CROP_POLLINATOR_DEPENDENCE_FIELD]

