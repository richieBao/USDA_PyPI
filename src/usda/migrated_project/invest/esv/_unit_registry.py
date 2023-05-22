import pint
import os
import usda
from importlib import resources

# the same unit registry instance should be shared across everything
# load from custom unit defintions file
# don't raise warnings when redefining units
u = pint.UnitRegistry(on_redefinition='ignore')

if __package__:
    module_name = usda.__name__
    unit_definitions_fn='migrated_project/invest/configs/unit_definitions.txt'
    ureg_config=resources.files(module_name).joinpath(unit_definitions_fn)     
    u.load_definitions(ureg_config)    
else:
    u.load_definitions('../configs/unit_definitions.txt')
    



