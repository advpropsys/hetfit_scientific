<!-- markdownlint-disable -->

<a href="../nets/envs.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `nets.envs`






---

<a href="../nets/envs.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SCI`




<a href="../nets/envs.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    hidden_dim: int = 200,
    dropout: bool = True,
    epochs: int = 10,
    dataset: str = 'test.pkl',
    sample_size: int = 1000,
    source: str = 'dataset.csv',
    boundary_conditions: list = None
)
```








---

<a href="../nets/envs.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    columns: tuple = None,
    idx: tuple = None,
    optim: <module 'optim' from '/Users/apsys/Downloads/hetfit2/venv/lib/10/site-packages/torch/optim/py'> = <class 'torch.optim.adamw.AdamW'>,
    loss: <module 'nn' from '/Users/apsys/Downloads/hetfit2/venv/lib/10/site-packages/torch/nn/py'> = <class 'torch.nn.modules.loss.L1Loss'>,
    model: Module = <class 'nets.deep_dense.dmodel'>,
    custom: bool = False
) → None
```

Builds model, loss, optimizer. Has defaults 

**Args:**
 
 - <b>`columns`</b> (tuple, optional):  Columns to be selected for feature fitting. Defaults to (1,3,3,5). optim - torch Optimizer loss - torch Loss function (nn) 

---

<a href="../nets/envs.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `data_flow`

```python
data_flow(
    columns_idx: tuple = (1, 3, 3, 5),
    idx: tuple = None,
    split_idx: int = 800
) → DataLoader
```

Data prep pipeline 



**Args:**
 
 - <b>`columns_idx`</b> (tuple, optional):  Columns to be selected (sliced 1:2 3:4) for feature fitting. Defaults to (1,3,3,5).  
 - <b>`idx`</b> (tuple, optional):  2|3 indexes to be selected for feature fitting. Defaults to None. Use either idx or columns_idx (for F:R->R idx, for F:R->R2 columns_idx) 
 - <b>`split_idx (int) `</b>:  Index to split for training 



**Returns:**
 
 - <b>`torch.utils.data.DataLoader`</b>:  Torch native dataloader 

---

<a href="../nets/envs.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `feature_gen`

```python
feature_gen(
    base: bool = True,
    fname: str = None,
    index: int = None,
    func=None
) → None
```





---

<a href="../nets/envs.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `feature_importance`

```python
feature_importance(X: DataFrame, Y: Series, verbose: int = 1)
```





---

<a href="../nets/envs.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inference`

```python
inference(
    X: <built-in method tensor of type object at 0x1080a6320>,
    model_name: str = None
) → ndarray
```

Inference of (pre-)trained model 



**Args:**
 
 - <b>`X`</b> (tensor):  your data in domain of train 



**Returns:**
 
 - <b>`np.ndarray`</b>:  predictions 

---

<a href="../nets/envs.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_seed`

```python
init_seed(seed)
```

Initializes seed for torch optional()  



---

<a href="../nets/envs.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `jit_export`

```python
jit_export(path: str = './models/model.pt')
```





---

<a href="../nets/envs.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `onnx_export`

```python
onnx_export(path: str = './models/model.onnx')
```





---

<a href="../nets/envs.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `performance`

```python
performance(c=0.4) → dict
```





---

<a href="../nets/envs.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `performance_super`

```python
performance_super(
    c=0.4,
    real_data_column_index: tuple = (1, 8),
    real_data_samples: int = 23,
    generated_length: int = 1000
) → dict
```





---

<a href="../nets/envs.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot`

```python
plot()
```





---

<a href="../nets/envs.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot3d`

```python
plot3d()
```





---

<a href="../nets/envs.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(name: str = 'model.pt') → None
```





---

<a href="../nets/envs.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train`

```python
train(epochs: int = 10) → None
```

Train model If sklearn instance uses .fit() 

---

<a href="../nets/envs.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train_epoch`

```python
train_epoch(X, model, loss_function, optim)
```






---

<a href="../nets/envs.py#L277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RCI`




<a href="../nets/envs.py#L278"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwargs)
```








---

<a href="../nets/envs.py#L329"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    columns: tuple = None,
    idx: tuple = (3, 1),
    optim: <module 'optim' from '/Users/apsys/Downloads/hetfit2/venv/lib/10/site-packages/torch/optim/py'> = <class 'torch.optim.adamw.AdamW'>,
    loss: <module 'nn' from '/Users/apsys/Downloads/hetfit2/venv/lib/10/site-packages/torch/nn/py'> = <class 'torch.nn.modules.loss.L1Loss'>,
    model: Module = <class 'PINN.pinns.PINNd_p'>,
    lr: float = 0.001
) → None
```

Builds model, loss, optimizer. Has defaults 

**Args:**
 
 - <b>`columns`</b> (tuple, optional):  Columns to be selected for feature fitting. Defaults to None. 
 - <b>`idx`</b> (tuple, optional):  indexes to be selected Default (3,1) optim - torch Optimizer loss - torch Loss function (nn) 

---

<a href="../nets/envs.py#L281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `data_flow`

```python
data_flow(
    columns_idx: tuple = (1, 3, 3, 5),
    idx: tuple = None,
    split_idx: int = 800
) → DataLoader
```

Data prep pipeline 



**Args:**
 
 - <b>`columns_idx`</b> (tuple, optional):  Columns to be selected (sliced 1:2 3:4) for feature fitting. Defaults to (1,3,3,5).  
 - <b>`idx`</b> (tuple, optional):  2|3 indexes to be selected for feature fitting. Defaults to None. Use either idx or columns_idx (for F:R->R idx, for F:R->R2 columns_idx) 
 - <b>`split_idx (int) `</b>:  Index to split for training 



**Returns:**
 
 - <b>`torch.utils.data.DataLoader`</b>:  Torch native dataloader 

---

<a href="../nets/envs.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `feature_gen`

```python
feature_gen(
    base: bool = True,
    fname: str = None,
    index: int = None,
    func=None
) → None
```





---

<a href="../nets/envs.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `feature_importance`

```python
feature_importance(X: DataFrame, Y: Series, verbose: int = 1)
```





---

<a href="../nets/envs.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inference`

```python
inference(
    X: <built-in method tensor of type object at 0x1080a6320>,
    model_name: str = None
) → ndarray
```

Inference of (pre-)trained model 



**Args:**
 
 - <b>`X`</b> (tensor):  your data in domain of train 



**Returns:**
 
 - <b>`np.ndarray`</b>:  predictions 

---

<a href="../nets/envs.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_seed`

```python
init_seed(seed)
```

Initializes seed for torch optional()  



---

<a href="../nets/envs.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `jit_export`

```python
jit_export(path: str = './models/model.pt')
```





---

<a href="../nets/envs.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `onnx_export`

```python
onnx_export(path: str = './models/model.onnx')
```





---

<a href="../nets/envs.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `performance`

```python
performance(c=0.4) → dict
```





---

<a href="../nets/envs.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `performance_super`

```python
performance_super(
    c=0.4,
    real_data_column_index: tuple = (1, 8),
    real_data_samples: int = 23,
    generated_length: int = 1000
) → dict
```





---

<a href="../nets/envs.py#L362"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot`

```python
plot()
```





---

<a href="../nets/envs.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot3d`

```python
plot3d()
```





---

<a href="../nets/envs.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(name: str = 'model.pt') → None
```





---

<a href="../nets/envs.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train`

```python
train(epochs: int = 10) → None
```

Train model If sklearn instance uses .fit() 

---

<a href="../nets/envs.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train_epoch`

```python
train_epoch(X, model, loss_function, optim)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
