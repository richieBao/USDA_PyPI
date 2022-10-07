# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:57:02 2022

@author: Richie Bao-caDesign设计(cadesign.cn)
"""
from setuptools import setup,find_packages,find_namespace_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='usda', #应用名，即包名
    version='0.1.1', #版本号
    license="MIT", #版权声明，BSD,MIT
    author='Richie Bao-caDesign设计(cadesign.cn)', #作者名
    author_email='richiebao@outlook.com', #作者邮箱
    description='USDA Urban Spatial Data Analysis Method', #描述
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://richiebao.github.io/USDA_CH_final',  #项目主页 
    package_dir={"": "src"},
    packages=find_packages(where='src'),#包括安装包内的python包；find_namespace_packages()，和find_packages() ['toolkit4beginner']
    python_requires='>=3.6', #pyton版本控制
    platforms='any',
    #install_requires=['matplotlib','statistics','numpy'] #自动安装依赖包（库）
    # include_package_data=True, #如果配置有MANIFEST.in，包含数据文件或额外其它文，该参数配置为True，则一同打包
    )