# pylis

The `pylis` library contains helpful functions for pre- and postprocessing for the NASA Land Information System in Python (e.g., reading in data, visualization, CDF matching, ...).

## How to use

Clone the repository from KUL-RSDA (or use your own fork if you want to make personal changes to the code):
```
cd your/path/to/scripts
git clone https://github.com/KUL-RSDA/pylis
```

When you are working on a Python script or notebook in which you want to use `pylis` functionalities, add these lines to the top:
```python 
# enable loading in the necessary scripts
# use here the parent directory where the pylis folder is located

import sys
sys.path.append("your/path/to/scripts")
```

You can then use `pylis` as if it is a library you have installed, e.g.,
```python
from pylis import readers
from pylis import visualization as vis
from pylis.help import root_zone

dc_sm = readers.lis_cube(...)
vis.map_imshow(root_zone(dc_sm).mean(dim = "time"))
```

For use cases on how to use the different functionalities, you can consult the notebooks under the `tutorials` folder.
