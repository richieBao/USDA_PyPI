# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 19:14:13 2023

@author: richie bao
"""
import planetary_computer
import pystac_client
import matplotlib

def worldcover_cmap4plot(boundary=(-1,51.15,1,52)):
    catalog=pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace,)
    search=catalog.search(collections=["esa-worldcover"],bbox=boundary)
    items=list(search.get_items())    
    
    class_list=items[0].assets["map"].extra_fields["classification:classes"]
    classmap={c["value"]: {"description": c["description"], "hex": c["color-hint"]} for c in class_list}

    colors = ["#000000" for r in range(256)]
    for key, value in classmap.items():
        colors[int(key)] = f"#{value['hex']}"
    cmap = matplotlib.colors.ListedColormap(colors)
    
    # sequences needed for an informative colorbar
    values = [key for key in classmap]
    boundaries = [(values[i + 1] + values[i]) / 2 for i in range(len(values) - 1)]
    boundaries = [0] + boundaries + [255]
    ticks = [(boundaries[i + 1] + boundaries[i]) / 2 for i in range(len(boundaries) - 1)]
    tick_labels = [value["description"] for value in classmap.values()]        
    
    
    return cmap,values,boundaries,ticks,tick_labels

if __name__=="__main__":
    worldcover_cmap4plot()