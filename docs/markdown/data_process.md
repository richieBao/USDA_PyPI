# usda.data_process

?> OSM数据处理

## data_process.shpPolygon2OsmosisTxt

转换shape的polygon为osmium的polygon数据格式（.txt），用于.osm地图数据的裁切

```python
shpPolygon2OsmosisTxt(shape_polygon_fp, osmosis_txt_fp)
    function - 转换shape的polygon为osmium的polygon数据格式（.txt），用于.osm地图数据的裁切
    
    Params:
        shape_polygon_fp - 输入shape地理数据格式的polygon文件路径；string
        osmosis_txt_fp - 输出为osmosis格式的polygon数据格式.txt文件路径；string
        
    Returns:
        None
```

## data_process.osmHandler

通过继承osmium类 class osmium.SimpleHandler读取.osm数据

```python
class osmHandler(osmium._osmium.SimpleHandler)
 |  class-通过继承osmium类 class osmium.SimpleHandler读取.osm数据. 
 |  
 |  代码示例：
 |  osm_Chicago_fp=r"F:\data\osm_clip.osm" 
 |  osm_handler=osmHandler() 
 |  osm_handler.apply_file(osm_Chicago_fp,locations=True)
 |  
 |  Method resolution order:
 |      osmHandler
 |      osmium._osmium.SimpleHandler
 |      osmium._osmium.BaseHandler
 |      pybind11_builtins.pybind11_object
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self)
 |      __init__(self: osmium._osmium.SimpleHandler) -> None
 |  
 |  area(self, a)
 |  
 |  node(self, n)
 |  
 |  way(self, w)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from osmium._osmium.SimpleHandler:
 |  
 |  apply_buffer(...)
 |      apply_buffer(self: osmium._osmium.SimpleHandler, buffer: buffer, format: str, locations: bool = False, idx: str = 'flex_mem') -> None
 |      
 |      Apply the handler to a string buffer. The buffer must be a
 |      byte string.
 |  
 |  apply_file(...)
 |      apply_file(self: osmium._osmium.SimpleHandler, filename: object, locations: bool = False, idx: str = 'flex_mem') -> None
 |      
 |      Apply the handler to the given file. If locations is true, then
 |      a location handler will be applied before, which saves the node
 |      positions. In that case, the type of this position index can be
 |      further selected in idx. If an area callback is implemented, then
 |      the file will be scanned twice and a location handler and a
 |      handler for assembling multipolygons and areas from ways will
 |      be executed.
 |  
 |  ----------------------------------------------------------------------
 |  Static methods inherited from pybind11_builtins.pybind11_object:
 |  
 |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
 |      Create and return a new object.  See help(type) for accurate signature.
```

## data_process.save_osm

根据条件逐个保存读取的osm数据（node, way and area）

```python
save_osm(osm_handler, osm_type, save_path='./data/', fileType='GPKG')
    function - 根据条件逐个保存读取的osm数据（node, way and area）
    
    Params:
        osm_handler - osm返回的node,way和area数据，配套类osmHandler(osm.SimpleHandler)实现；Class
        osm_type - 要保存的osm元素类型，包括"node"，"way"和"area"；string
        save_path - 保存路径。The default is "./data/"；string
        fileType - 保存的数据类型，包括"shp", "GeoJSON", "GPKG"。The default is "GPKG"；string
    
    Returns:
        osm_node_gdf - OSM的node类；GeoDataFrame(GeoPandas)
        osm_way_gdf - OSM的way类；GeoDataFrame(GeoPandas)
        osm_area_gdf - OSM的area类；GeoDataFrame(GeoPandas)
```

?> 地理信息格式数据处理

## data_process.pts2raster

将.shp格式的点数据转换为.tif栅格(raster)

```python
pts2raster(pts_shp, raster_path, cellSize, field_name=False)
    function - 将.shp格式的点数据转换为.tif栅格(raster)
               将点数据写入为raster数据。使用raster.SetGeoTransform，栅格化数据。参考GDAL官网代码 
    
    Params:
        pts_shp - .shp格式点数据文件路径；SHP点数据
        raster_path - 保存的栅格文件路径；string
        cellSize - 栅格单元大小；int
        field_name - 写入栅格的.shp点数据属性字段；string
        
    Returns:
        返回读取已经保存的栅格数据；array
```

