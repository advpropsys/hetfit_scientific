<!-- markdownlint-disable -->

<a href="../utils/dataset_loader.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `utils.dataset_loader`





---

<a href="../utils/dataset_loader.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_dataset`

```python
get_dataset(
    raw: bool = False,
    sample_size: int = 1000,
    name: str = 'dataset.pkl',
    source: str = 'dataset.csv',
    boundary_conditions: list = None
) → <module '_pickle' from '/opt/homebrew/Cellar/python@10/9/Frameworks/framework/Versions/10/lib/10/lib-dynload/cpython-310-so'>
```

Gets augmented dataset 



**Args:**
 
 - <b>`raw`</b> (bool, optional):  either to use source data or augmented. Defaults to False. 
 - <b>`sample_size`</b> (int, optional):  sample size. Defaults to 1000. 
 - <b>`name`</b> (str, optional):  name of wanted dataset. Defaults to 'dataset.pkl'. 
 - <b>`boundary_conditions`</b> (list,optional):  y1,y2,x1,x2. 

**Returns:**
 
 - <b>`_pickle`</b>:  pickle buffer 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
