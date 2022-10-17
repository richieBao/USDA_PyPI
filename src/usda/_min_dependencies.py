# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 08:29:05 2022

@author: richie bao
"""
from collections import defaultdict

"""All minimum dependencies for scikit-learn."""
# import platform
# print(platform.python_implementation())
dependent_packages={
    "tqdm":("4.64.1","docs")   
    
    }

# create inverse mapping for setuptools
tag_to_packages: dict=defaultdict(list)
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))
        
        
        

# Used by CI to get the min dependencies
if __name__ == "__main__":
    print(sorted(tag_to_packages.items()))


