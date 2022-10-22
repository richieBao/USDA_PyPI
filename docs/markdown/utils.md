# usda.utils

## utils.Bunch

构建类属性“字典”

> Code Migrating from: scikit-learn/sklearn/utils/_bunch.py

```python
class Bunch(builtins.dict)
 |  Bunch(**kwargs)
 |  
 |  Container object exposing keys as attributes.
 |  Bunch objects are sometimes used as an output for functions and methods.
 |  They extend dictionaries by enabling values to be accessed by key,
 |  `bunch["value_key"]`, or by an attribute, `bunch.value_key`.
 |  Examples
 |  -------- 
 |  Methods defined here: 
 |  clear(...)
 |      D.clear() -> None.  Remove all items from D.
 |  
 |  copy(...)
 |      D.copy() -> a shallow copy of D
 |  
 |  get(self, key, default=None, /)
 |      Return the value for key if key is in the dictionary, else default.
 |  
 |  items(...)
 |      D.items() -> a set-like object providing a view on D's items
 |  
 |  keys(...)
 |      D.keys() -> a set-like object providing a view on D's keys
 |  
 |  pop(...)
 |      D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
 |      If key is not found, d is returned if given, otherwise KeyError is raised
 |  
 |  popitem(self, /)
 |      Remove and return a (key, value) pair as a 2-tuple.
 |      
 |      Pairs are returned in LIFO (last-in, first-out) order.
 |      Raises KeyError if the dict is empty.
 |  
 |  setdefault(self, key, default=None, /)
 |      Insert key with a value of default if key is not in the dictionary.
 |      
 |      Return the value for key if key is in the dictionary, else default.
 |  
 |  update(...)
 |      D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
 |      If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
 |      If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
 |      In either case, this is followed by: for k in F:  D[k] = F[k]
 |  
 |  values(...)
 |      D.values() -> an object providing a view on D's values
```

* __Examples__

```python
>>> from usda.utils import Bunch
>>> b = Bunch(a=1, b=2)
>>> b['b']
2
>>> b.b
2
>>> b.a = 3
>>> b['a']
3
>>> b.c = 6
>>> b['c']
6
```

## utils.DisplayablePath

返回指定路径下所有文件夹及其下文件的结构

> Code Migrating from: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python

```python
class DisplayablePath(builtins.object)
 |  DisplayablePath(path, parent_path, is_last)
 |  
 |  class - 返回指定路径下所有文件夹及其下文件的结构
 ```

* __Examples__

```python
from pathlib import Path
app_root=r'C:\Users\richi\Pictures'    
paths = utils.DisplayablePath.make_tree(Path(app_root))
for path in paths:
    print(path.displayable())  
```

```
Pictures/
├── 2_1_03_07.jpg
├── 3_2_24.jpg
├── 3_2_24.png
├── 640.jpg
├── background.jpg
├── background.psd
├── caDesign.png
├── Camera Roll/
│   └── desktop.ini
├── Debut/
├── desktop.ini
├── qgis_style_pts_01.qml
├── QQ截图20220320081229.png
├── QQ截图20220419080451.png
├── QQ截图20220419083716.png
├── QQ截图20220419090125.png
└── Saved Pictures/
    └── desktop.ini
```

## utils.filePath_extraction

以所在文件夹路径为键，值为包含该文件夹下所有文件名的列表。文件类型可以自行定义 

```python
filePath_extraction(dirpath, fileType)
    funciton  - 以所在文件夹路径为键，值为包含该文件夹下所有文件名的列表。文件类型可以自行定义 
    
    Params:
        dirpath - 根目录，存储所有待读取的文件；string
        fileType - 待读取文件的类型；list(string)
        
    Returns:
        filePath_Info - 文件路径字典，文件夹路径为键，文件夹下的文件名列表为值；dict
```

## utils.start_time

计算当前时间

```python
start_time()
    function-计算当前时间
```

## utils.duration

计算持续时间

```python
duration(start_time)
    function-计算持续时间
    
    Params:
    start_time - 开始时间；datatime
```