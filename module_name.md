# Table of Contents

* [main](#main)
* [PINN](#PINN)
* [PINN.pinns](#PINN.pinns)
  * [PINNd\_p](#PINN.pinns.PINNd_p)
  * [PINNhd\_ma](#PINN.pinns.PINNhd_ma)
  * [PINNT\_ma](#PINN.pinns.PINNT_ma)
* [utils](#utils)
* [utils.test](#utils.test)
* [utils.dataset\_loader](#utils.dataset_loader)
  * [get\_dataset](#utils.dataset_loader.get_dataset)
* [utils.ndgan](#utils.ndgan)
  * [DCGAN](#utils.ndgan.DCGAN)
    * [define\_discriminator](#utils.ndgan.DCGAN.define_discriminator)
    * [generate\_latent\_points](#utils.ndgan.DCGAN.generate_latent_points)
    * [define\_gan](#utils.ndgan.DCGAN.define_gan)
    * [summarize\_performance](#utils.ndgan.DCGAN.summarize_performance)
    * [train\_gan](#utils.ndgan.DCGAN.train_gan)
* [utils.data\_augmentation](#utils.data_augmentation)
  * [dataset](#utils.data_augmentation.dataset)
    * [\_\_init\_\_](#utils.data_augmentation.dataset.__init__)
* [nets](#nets)
* [nets.envs](#nets.envs)
  * [SCI](#nets.envs.SCI)
    * [data\_flow](#nets.envs.SCI.data_flow)
    * [init\_seed](#nets.envs.SCI.init_seed)
    * [compile](#nets.envs.SCI.compile)
    * [train](#nets.envs.SCI.train)
    * [inference](#nets.envs.SCI.inference)
  * [RCI](#nets.envs.RCI)
    * [data\_flow](#nets.envs.RCI.data_flow)
    * [compile](#nets.envs.RCI.compile)
* [nets.dense](#nets.dense)
  * [Net](#nets.dense.Net)
    * [\_\_init\_\_](#nets.dense.Net.__init__)
* [nets.design](#nets.design)
  * [B\_field\_norm](#nets.design.B_field_norm)
* [nets.deep\_dense](#nets.deep_dense)
  * [dmodel](#nets.deep_dense.dmodel)
    * [\_\_init\_\_](#nets.deep_dense.dmodel.__init__)

<a id="main"></a>

# main

<a id="PINN"></a>

# PINN

<a id="PINN.pinns"></a>

# PINN.pinns

<a id="PINN.pinns.PINNd_p"></a>

## PINNd\_p Objects

```python
class PINNd_p(nn.Module)
```

$d \mapsto P$

<a id="PINN.pinns.PINNhd_ma"></a>

## PINNhd\_ma Objects

```python
class PINNhd_ma(nn.Module)
```

$h,d \mapsto m_a $

<a id="PINN.pinns.PINNT_ma"></a>

## PINNT\_ma Objects

```python
class PINNT_ma(nn.Module)
```

$ m_a, U \mapsto T$

<a id="utils"></a>

# utils

<a id="utils.test"></a>

# utils.test

<a id="utils.dataset_loader"></a>

# utils.dataset\_loader

<a id="utils.dataset_loader.get_dataset"></a>

#### get\_dataset

```python
def get_dataset(raw: bool = False,
                sample_size: int = 1000,
                name: str = 'dataset.pkl',
                source: str = 'dataset.csv',
                boundary_conditions: list = None) -> _pickle
```

Gets augmented dataset

**Arguments**:

- `raw` _bool, optional_ - either to use source data or augmented. Defaults to False.
- `sample_size` _int, optional_ - sample size. Defaults to 1000.
- `name` _str, optional_ - name of wanted dataset. Defaults to 'dataset.pkl'.
- `boundary_conditions` _list,optional_ - y1,y2,x1,x2.

**Returns**:

- `_pickle` - pickle buffer

<a id="utils.ndgan"></a>

# utils.ndgan

<a id="utils.ndgan.DCGAN"></a>

## DCGAN Objects

```python
class DCGAN()
```

<a id="utils.ndgan.DCGAN.define_discriminator"></a>

#### define\_discriminator

```python
def define_discriminator(inputs=8)
```

function to return the compiled discriminator model

<a id="utils.ndgan.DCGAN.generate_latent_points"></a>

#### generate\_latent\_points

```python
def generate_latent_points(latent_dim, n)
```

generate points in latent space as input for the generator

<a id="utils.ndgan.DCGAN.define_gan"></a>

#### define\_gan

```python
def define_gan(generator, discriminator)
```

define the combined generator and discriminator model

<a id="utils.ndgan.DCGAN.summarize_performance"></a>

#### summarize\_performance

```python
def summarize_performance(epoch, generator, discriminator, latent_dim, n=200)
```

evaluate the discriminator and plot real and fake samples

<a id="utils.ndgan.DCGAN.train_gan"></a>

#### train\_gan

```python
def train_gan(g_model,
              d_model,
              gan_model,
              latent_dim,
              num_epochs=2500,
              num_eval=2500,
              batch_size=2)
```

function to train gan model

<a id="utils.data_augmentation"></a>

# utils.data\_augmentation

<a id="utils.data_augmentation.dataset"></a>

## dataset Objects

```python
class dataset()
```

Creates dataset from input source

<a id="utils.data_augmentation.dataset.__init__"></a>

#### \_\_init\_\_

```python
def __init__(number_samples: int,
             name: str,
             source: str,
             boundary_conditions: list = None)
```

_summary_

**Arguments**:

- `number_samples` _int_ - _description_
- `name` _str_ - _description_
- `source` _str_ - _description_
- `boundary_conditions` _list_ - y1,y2,x1,x2

<a id="nets"></a>

# nets

<a id="nets.envs"></a>

# nets.envs

<a id="nets.envs.SCI"></a>

## SCI Objects

```python
class SCI()
```

<a id="nets.envs.SCI.data_flow"></a>

#### data\_flow

```python
def data_flow(columns_idx: tuple = (1, 3, 3, 5),
              idx: tuple = None,
              split_idx: int = 800) -> torch.utils.data.DataLoader
```

Data prep pipeline

**Arguments**:

- `columns_idx` _tuple, optional_ - Columns to be selected (sliced 1:2 3:4) for feature fitting. Defaults to (1,3,3,5).
- `idx` _tuple, optional_ - 2|3 indexes to be selected for feature fitting. Defaults to None. Use either idx or columns_idx (for F:R->R idx, for F:R->R2 columns_idx)
  split_idx (int) : Index to split for training
  

**Returns**:

- `torch.utils.data.DataLoader` - Torch native dataloader

<a id="nets.envs.SCI.init_seed"></a>

#### init\_seed

```python
def init_seed(seed)
```

Initializes seed for torch optional()

<a id="nets.envs.SCI.compile"></a>

#### compile

```python
def compile(columns: tuple = None,
            idx: tuple = None,
            optim: torch.optim = torch.optim.AdamW,
            loss: nn = nn.L1Loss,
            model: nn.Module = dmodel,
            custom: bool = False) -> None
```

Builds model, loss, optimizer. Has defaults

**Arguments**:

- `columns` _tuple, optional_ - Columns to be selected for feature fitting. Defaults to (1,3,3,5).
  optim - torch Optimizer
  loss - torch Loss function (nn)

<a id="nets.envs.SCI.train"></a>

#### train

```python
def train(epochs: int = 10) -> None
```

Train model
If sklearn instance uses .fit()

<a id="nets.envs.SCI.inference"></a>

#### inference

```python
def inference(X: tensor, model_name: str = None) -> np.ndarray
```

Inference of (pre-)trained model

**Arguments**:

- `X` _tensor_ - your data in domain of train
  

**Returns**:

- `np.ndarray` - predictions

<a id="nets.envs.RCI"></a>

## RCI Objects

```python
class RCI(SCI)
```

<a id="nets.envs.RCI.data_flow"></a>

#### data\_flow

```python
def data_flow(columns_idx: tuple = (1, 3, 3, 5),
              idx: tuple = None,
              split_idx: int = 800) -> torch.utils.data.DataLoader
```

Data prep pipeline

**Arguments**:

- `columns_idx` _tuple, optional_ - Columns to be selected (sliced 1:2 3:4) for feature fitting. Defaults to (1,3,3,5).
- `idx` _tuple, optional_ - 2|3 indexes to be selected for feature fitting. Defaults to None. Use either idx or columns_idx (for F:R->R idx, for F:R->R2 columns_idx)
  split_idx (int) : Index to split for training
  

**Returns**:

- `torch.utils.data.DataLoader` - Torch native dataloader

<a id="nets.envs.RCI.compile"></a>

#### compile

```python
def compile(columns: tuple = None,
            idx: tuple = (3, 1),
            optim: torch.optim = torch.optim.AdamW,
            loss: nn = nn.L1Loss,
            model: nn.Module = PINNd_p,
            lr: float = 0.001) -> None
```

Builds model, loss, optimizer. Has defaults

**Arguments**:

- `columns` _tuple, optional_ - Columns to be selected for feature fitting. Defaults to None.
- `idx` _tuple, optional_ - indexes to be selected Default (3,1)
  optim - torch Optimizer
  loss - torch Loss function (nn)

<a id="nets.dense"></a>

# nets.dense

<a id="nets.dense.Net"></a>

## Net Objects

```python
class Net(nn.Module)
```

4 layer model, different activations and neurons count on layer

<a id="nets.dense.Net.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_dim: int = 2, hidden_dim: int = 200)
```

Init

**Arguments**:

- `input_dim` _int, optional_ - Defaults to 2.
- `hidden_dim` _int, optional_ - Defaults to 200.

<a id="nets.design"></a>

# nets.design

<a id="nets.design.B_field_norm"></a>

#### B\_field\_norm

```python
def B_field_norm(Bmax, L, k=16, plot=True)
```

Returns vec B_z

**Arguments**:

- `Bmax` _any_ - maximum B in thruster
  k - magnetic field profile number

<a id="nets.deep_dense"></a>

# nets.deep\_dense

<a id="nets.deep_dense.dmodel"></a>

## dmodel Objects

```python
class dmodel(nn.Module)
```

4 layers Torch model. Relu activations, hidden layers are same size.

<a id="nets.deep_dense.dmodel.__init__"></a>

#### \_\_init\_\_

```python
def __init__(in_features=1, hidden_features=200, out_features=1)
```

Init

**Arguments**:

- `in_features` _int, optional_ - Input features. Defaults to 1.
- `hidden_features` _int, optional_ - Hidden dims. Defaults to 200.
- `out_features` _int, optional_ - Output dims. Defaults to 1.

