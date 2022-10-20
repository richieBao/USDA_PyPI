# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:57:02 2022

@author: Richie Bao-caDesign设计(cadesign.cn)

python setup.py sdist
twine upload dist/usda-0.0.14.tar.gz
"""
from setuptools import setup,find_packages,find_namespace_packages
import src.usda._min_dependencies as min_deps 

DISTNAME="usda"
MAINTAINER="Richie Bao"
MAINTAINER_EMAIL="richiebao@outlook.com"
DESCRIPTION="A set of python modules for Urban Spatial Data Analysis Method (USDA)"
LICENSE="new BSD"
URL="https://richiebao.github.io/USDA_PyPI"
DOWNLOAD_URL="https://github.com/richieBao/USDA_PyPI"
PROJECT_URLS={
    # "Bug Tracker":"",
    "Documentation":"https://richiebao.github.io/USDA_PyPI",
    "Source Code": "https://github.com/richieBao/USDA_PyPI"
    }

import src.usda
VERSION=src.usda.__version__

with open("README.rst") as f:
    LONG_DESCRIPTION=f.read()
    
extra_setuptools_args=dict(
    zip_safe=False,  # the package can run out of an .egg file
    include_package_data=True,
    extras_require={key: min_deps.tag_to_packages[key] for key in ["examples", "docs", "tests", "benchmark"]},
    )    

PACKAGE_DATA={"usda.datasets.data":["*.pickle"]}

def setup_package():
    python_requires=">=3.7"
    
    metadata=dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Development Status :: 5 - Production/Stable",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: Implementation :: CPython",
            "Programming Language :: Python :: Implementation :: PyPy",
        ],        
        # cmdclass=cmdclass,
        python_requires=python_requires,
        install_requires=min_deps.tag_to_packages["install"],
        # package_data={"": ["*.pxd"]},
        **extra_setuptools_args,
        # include_package_data=True,
        package_data=PACKAGE_DATA,
        )    
    from setuptools import setup
    
    PACKAGES=find_packages(where='src')
    metadata["packages"]=PACKAGES
    metadata['package_dir']={"": "src"} 
    
    # print(metadata)
    setup(**metadata)

'''
setup(name='usda', #应用名，即包名
    version='0.0.1', #版本号
    license="MIT", #版权声明，BSD,MIT
    author='Richie Bao-caDesign设计(cadesign.cn)', #作者名
    author_email='richiebao@outlook.com', #作者邮箱
    description='USDA Urban Spatial Data Analysis Method', #描述
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url='https://richiebao.github.io/USDA_CH_final',  #项目主页 
    package_dir={"": "src"},
    packages=find_packages(where='src'),#包括安装包内的python包；find_namespace_packages()，和find_packages() ['toolkit4beginner']
    python_requires='>=3.6', #pyton版本控制
    platforms='any',
    #install_requires=['matplotlib','statistics','numpy'] #自动安装依赖包（库）
    # include_package_data=True, #如果配置有MANIFEST.in，包含数据文件或额外其它文，该参数配置为True，则一同打包
    )
'''

if __name__=="__main__":
    # print(extra_setuptools_args)
    setup_package()