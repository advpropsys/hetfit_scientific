# Table of Contents

- [Table of Contents](#table-of-contents)
- [main](#main)
- [PINN](#pinn)
- [PINN.pinns](#pinnpinns)
  - [PINNd\_p Objects](#pinnd_p-objects)
  - [PINNhd\_ma Objects](#pinnhd_ma-objects)
  - [PINNT\_ma Objects](#pinnt_ma-objects)
- [utils](#utils)
- [utils.test](#utilstest)
- [utils.dataset\_loader](#utilsdataset_loader)
      - [get\_dataset](#get_dataset)
- [utils.ndgan](#utilsndgan)
  - [DCGAN Objects](#dcgan-objects)
      - [define\_discriminator](#define_discriminator)
      - [generate\_latent\_points](#generate_latent_points)
      - [define\_gan](#define_gan)
      - [summarize\_performance](#summarize_performance)
      - [train\_gan](#train_gan)
- [utils.data\_augmentation](#utilsdata_augmentation)
  - [dataset Objects](#dataset-objects)
      - [\_\_init\_\_](#__init__)
- [nets](#nets)
- [nets.envs](#netsenvs)
  - [SCI Objects](#sci-objects)
      - [\_\_init\_\_](#__init__-1)
      - [feature\_gen](#feature_gen)
      - [feature\_importance](#feature_importance)
      - [data\_flow](#data_flow)
      - [init\_seed](#init_seed)
      - [compile](#compile)
      - [train](#train)
      - [jit\_export](#jit_export)
      - [inference](#inference)
      - [plot](#plot)
      - [plot3d](#plot3d)
      - [performance](#performance)
      - [performance\_super](#performance_super)
  - [RCI Objects](#rci-objects)
      - [data\_flow](#data_flow-1)
      - [compile](#compile-1)
      - [plot](#plot-1)
      - [performance](#performance-1)
- [nets.dense](#netsdense)
  - [Net Objects](#net-objects)
      - [\_\_init\_\_](#__init__-2)
- [nets.normflows](#netsnormflows)
  - [GaussianKDE Objects](#gaussiankde-objects)
      - [\_\_init\_\_](#__init__-3)
      - [boundary\_distribution](#boundary_distribution)
      - [sample](#sample)
      - [score\_samples](#score_samples)
  - [Parameters](#parameters)
  - [Returns](#returns)
      - [log\_prob](#log_prob)
  - [Parameters](#parameters-1)
  - [Returns](#returns-1)
  - [Chi2KDE Objects](#chi2kde-objects)
      - [\_\_init\_\_](#__init__-4)
      - [sample](#sample-1)
      - [score\_samples](#score_samples-1)
  - [Parameters](#parameters-2)
  - [Returns](#returns-2)
      - [log\_prob](#log_prob-1)
  - [Parameters](#parameters-3)
  - [Returns](#returns-3)
  - [PlanarFlow Objects](#planarflow-objects)
      - [h\_prime](#h_prime)
  - [NormalizingFlow Objects](#normalizingflow-objects)
      - [\_\_init\_\_](#__init__-5)
      - [sample](#sample-2)
      - [forward](#forward)
      - [random\_normal\_samples](#random_normal_samples)
  - [nflow Objects](#nflow-objects)
      - [\_\_init\_\_](#__init__-6)
      - [compile](#compile-2)
      - [train](#train-1)
      - [performance](#performance-2)
- [nets.design](#netsdesign)
      - [B\_field\_norm](#b_field_norm)
      - [PUdesign](#pudesign)
- [nets.deep\_dense](#netsdeep_dense)
  - [dmodel Objects](#dmodel-objects)
      - [\_\_init\_\_](#__init__-7)
- [nets.opti](#netsopti)
- [nets.opti.blackbox](#netsoptiblackbox)
  - [Hyper Objects](#hyper-objects)
      - [start\_study](#start_study)

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

Init

**Arguments**:

- `number_samples` _int_ - number of samples to be genarated
- `name` _str_ - name of dataset
- `source` _str_ - source file
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

Scaled computing interface.

**Arguments**:

- `hidden_dim` _int, optional_ - Max demension of hidden linear layer. Defaults to 200. Should be >80 in not 1d case
- `dropout` _bool, optional_ - LEGACY, don't use. Defaults to True.
- `epochs` _int, optional_ - Optionally specify epochs here, but better in train. Defaults to 10.
- `dataset` _str, optional_ - dataset to be selected from ./data. Defaults to 'test.pkl'. If name not exists, code will generate new dataset with upcoming parameters.
- `sample_size` _int, optional_ - Samples to be generated (note: BEFORE applying boundary conditions). Defaults to 1000.
- `source` _str, optional_ - Source from which data will be generated. Better to not change. Defaults to 'dataset.csv'.
- `boundary_conditions` _list, optional_ - If sepcified, whole dataset will be cut rectangulary. Input list is [ymin,ymax,xmin,xmax] type. Defaults to None.

<a id="nets.envs.SCI.__init__"></a>

#### \_\_init\_\_

```python
def __init__(hidden_dim: int = 200,
             dropout: bool = True,
             epochs: int = 10,
             dataset: str = 'test.pkl',
             sample_size: int = 1000,
             source: str = 'dataset.csv',
             boundary_conditions: list = None)
```

Init

**Arguments**:

- `hidden_dim` _int, optional_ - Max demension of hidden linear layer. Defaults to 200. Should be >80 in not 1d case
- `dropout` _bool, optional_ - LEGACY, don't use. Defaults to True.
- `epochs` _int, optional_ - Optionally specify epochs here, but better in train. Defaults to 10.
- `dataset` _str, optional_ - dataset to be selected from ./data. Defaults to 'test.pkl'. If name not exists, code will generate new dataset with upcoming parameters.
- `sample_size` _int, optional_ - Samples to be generated (note: BEFORE applying boundary conditions). Defaults to 1000.
- `source` _str, optional_ - Source from which data will be generated. Better to not change. Defaults to 'dataset.csv'.
- `boundary_conditions` _list, optional_ - If sepcified, whole dataset will be cut rectangulary. Input list is [ymin,ymax,xmin,xmax] type. Defaults to None.

<a id="nets.envs.SCI.feature_gen"></a>

#### feature\_gen

```python
def feature_gen(base: bool = True,
                fname: str = None,
                index: int = None,
                func=None) -> None
```

Generate new features. If base true, generates most obvious ones. You can customize this by adding
new feature as name of column - fname, index of parent column, and lambda function which needs to be applied elementwise.

**Arguments**:

- `base` _bool, optional_ - Defaults to True.
- `fname` _str, optional_ - Name of new column. Defaults to None.
- `index` _int, optional_ - Index of parent column. Defaults to None.
- `func` __type_, optional_ - lambda function. Defaults to None.

<a id="nets.envs.SCI.feature_importance"></a>

#### feature\_importance

```python
def feature_importance(X: pd.DataFrame, Y: pd.Series, verbose: int = 1)
```

Gets feature importance by SGD regression and score selection. Default threshold is 1.25*mean
input X as self.df.iloc[:,(columns of choice)]
Y as self.df.iloc[:,(column of choice)]

**Arguments**:

- `X` _pd.DataFrame_ - Builtin DataFrame
- `Y` _pd.Series_ - Builtin Series
- `verbose` _int, optional_ - either to or to not print actual report. Defaults to 1.

**Returns**:

  Report (str)

<a id="nets.envs.SCI.data_flow"></a>

#### data\_flow

```python
def data_flow(columns_idx: tuple = (1, 3, 3, 5),
              idx: tuple = None,
              split_idx: int = 800) -> torch.utils.data.DataLoader
```

Data prep pipeline
It is called automatically, don't call it in your code.

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
            custom: bool = False,
            lr: float = 0.0001) -> None
```

Builds model, loss, optimizer. Has defaults

**Arguments**:

- `columns` _tuple, optional_ - Columns to be selected for feature fitting. Defaults to (1,3,3,5).
  optim - torch Optimizer. Default AdamW
  loss - torch Loss function (nn). Defaults to L1Loss

<a id="nets.envs.SCI.train"></a>

#### train

```python
def train(epochs: int = 10) -> None
```

Train model
If sklearn instance uses .fit()

epochs - optional

<a id="nets.envs.SCI.jit_export"></a>

#### jit\_export

```python
def jit_export(path: str = './models/model.pt')
```

Exports properly defined model to jit

**Arguments**:

- `path` _str, optional_ - path to models. Defaults to './models/model.pt'.

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

<a id="nets.envs.SCI.plot"></a>

#### plot

```python
def plot()
```

Automatic 2d plot

<a id="nets.envs.SCI.plot3d"></a>

#### plot3d

```python
def plot3d(colX=0, colY=1)
```

Plot of inputs and predicted data in mesh format

**Returns**:

  plotly plot

<a id="nets.envs.SCI.performance"></a>

#### performance

```python
def performance(c=0.4) -> dict
```

Automatic APE based performance if applicable, else returns nan

**Arguments**:

- `c` _float, optional_ - ZDE mitigation constant. Defaults to 0.4.

**Returns**:

- `dict` - {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}

<a id="nets.envs.SCI.performance_super"></a>

#### performance\_super

```python
def performance_super(c=0.4,
                      real_data_column_index: tuple = (1, 8),
                      real_data_samples: int = 23,
                      generated_length: int = 1000) -> dict
```

Performance by custom parameters. APE loss

**Arguments**:

- `c` _float, optional_ - ZDE mitigation constant. Defaults to 0.4.
- `real_data_column_index` _tuple, optional_ - Defaults to (1,8).
- `real_data_samples` _int, optional_ - Defaults to 23.
- `generated_length` _int, optional_ - Defaults to 1000.

**Returns**:

- `dict` - {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}

<a id="nets.envs.RCI"></a>

## RCI Objects

```python
class RCI(SCI)
```

Real values interface, uses different types of NN, NO scaling.
Parent:
    SCI()

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

<a id="nets.envs.RCI.plot"></a>

#### plot

```python
def plot()
```

Plots 2d plot of prediction vs real values

<a id="nets.envs.RCI.performance"></a>

#### performance

```python
def performance(c=0.4) -> dict
```

RCI performnace. APE errors.

**Arguments**:

- `c` _float, optional_ - correction constant to mitigate division by 0 error. Defaults to 0.4.

**Returns**:

- `dict` - {'Generator_Accuracy, %':np.mean(a),'APE_abs, %':abs_ape,'Model_APE, %': ape}

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

<a id="nets.normflows"></a>

# nets.normflows

<a id="nets.normflows.GaussianKDE"></a>

## GaussianKDE Objects

```python
class GaussianKDE(Distribution)
```

<a id="nets.normflows.GaussianKDE.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X, bw)
```

X : tensor (n, d)
  `n` points with `d` dimensions to which KDE will be fit
bw : numeric
  bandwidth for Gaussian kernel

<a id="nets.normflows.GaussianKDE.boundary_distribution"></a>

#### boundary\_distribution

```python
def boundary_distribution(x1)
```

The function boundary_distribution takes in a value x1 and returns a value y1

**Arguments**:

- `x1`: the x-coordinate of the point on the boundary

**Returns**:

the y1 value.

<a id="nets.normflows.GaussianKDE.sample"></a>

#### sample

```python
def sample(num_samples)
```

We are sampling from a normal distribution with mean equal to the data points in the dataset and

standard deviation equal to the bandwidth

**Arguments**:

- `num_samples`: the number of samples to draw from the KDE

**Returns**:

A sample of size num_samples from the KDE.

<a id="nets.normflows.GaussianKDE.score_samples"></a>

#### score\_samples

```python
def score_samples(Y, X=None)
```

Returns the kernel density estimates of each point in `Y`.

Parameters
----------
Y : tensor (m, d)
  `m` points with `d` dimensions for which the probability density will
  be calculated
X : tensor (n, d), optional
  `n` points with `d` dimensions to which KDE will be fit. Provided to
  allow batch calculations in `log_prob`. By default, `X` is None and
  all points used to initialize KernelDensityEstimator are included.


Returns
-------
log_probs : tensor (m)
  log probability densities for each of the queried points in `Y`

<a id="nets.normflows.GaussianKDE.log_prob"></a>

#### log\_prob

```python
def log_prob(Y)
```

Returns the total log probability of one or more points, `Y`, using
a Multivariate Normal kernel fit to `X` and scaled using `bw`.

Parameters
----------
Y : tensor (m, d)
  `m` points with `d` dimensions for which the probability density will
  be calculated

Returns
-------
log_prob : numeric
  total log probability density for the queried points, `Y`

<a id="nets.normflows.Chi2KDE"></a>

## Chi2KDE Objects

```python
class Chi2KDE(Distribution)
```

<a id="nets.normflows.Chi2KDE.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X, bw)
```

X : tensor (n, d)
  `n` points with `d` dimensions to which KDE will be fit
bw : numeric
  bandwidth for Gaussian kernel

<a id="nets.normflows.Chi2KDE.sample"></a>

#### sample

```python
def sample(num_samples)
```

We are sampling from a log-normal distribution with a mean of the data points and a standard

deviation of the bandwidth

**Arguments**:

- `num_samples`: the number of samples to draw from the KDE

**Returns**:

A sample of size num_samples from the KDE.

<a id="nets.normflows.Chi2KDE.score_samples"></a>

#### score\_samples

```python
def score_samples(Y, X=None)
```

Returns the kernel density estimates of each point in `Y`.

Parameters
----------
Y : tensor (m, d)
  `m` points with `d` dimensions for which the probability density will
  be calculated
X : tensor (n, d), optional
  `n` points with `d` dimensions to which KDE will be fit. Provided to
  allow batch calculations in `log_prob`. By default, `X` is None and
  all points used to initialize KernelDensityEstimator are included.


Returns
-------
log_probs : tensor (m)
  log probability densities for each of the queried points in `Y`

<a id="nets.normflows.Chi2KDE.log_prob"></a>

#### log\_prob

```python
def log_prob(Y)
```

Returns the total log probability of one or more points, `Y`, using
a Multivariate Normal kernel fit to `X` and scaled using `bw`.

Parameters
----------
Y : tensor (m, d)
  `m` points with `d` dimensions for which the probability density will
  be calculated

Returns
-------
log_prob : numeric
  total log probability density for the queried points, `Y`

<a id="nets.normflows.PlanarFlow"></a>

## PlanarFlow Objects

```python
class PlanarFlow(nn.Module)
```

A single planar flow, computes T(x) and log(det(jac_T)))

<a id="nets.normflows.PlanarFlow.h_prime"></a>

#### h\_prime

```python
def h_prime(x)
```

Derivative of tanh

<a id="nets.normflows.NormalizingFlow"></a>

## NormalizingFlow Objects

```python
class NormalizingFlow(nn.Module)
```

A normalizng flow composed of a sequence of planar flows.

<a id="nets.normflows.NormalizingFlow.__init__"></a>

#### \_\_init\_\_

```python
def __init__(D, n_flows=2)
```

The function takes in two arguments, D and n_flows. D is the dimension of the data, and n_flows

is the number of flows. The function then creates a list of PlanarFlow objects, where the number
of PlanarFlow objects is equal to n_flows

**Arguments**:

- `D`: the dimensionality of the data
- `n_flows`: number of flows to use, defaults to 2 (optional)

<a id="nets.normflows.NormalizingFlow.sample"></a>

#### sample

```python
def sample(base_samples)
```

Transform samples from a simple base distribution
by passing them through a sequence of Planar flows.

<a id="nets.normflows.NormalizingFlow.forward"></a>

#### forward

```python
def forward(x)
```

Computes and returns the sum of log_det_jacobians
and the transformed samples T(x).

<a id="nets.normflows.random_normal_samples"></a>

#### random\_normal\_samples

```python
def random_normal_samples(n, dim=2)
```

It returns a tensor of size `n` by `dim` with random samples from a normal distribution with mean 0

and standard deviation 1.5

**Arguments**:

- `n`: number of samples
- `dim`: the dimension of the space we're sampling from, defaults to 2 (optional)

**Returns**:

A tensor of size n x dim with random normal samples.

<a id="nets.normflows.nflow"></a>

## nflow Objects

```python
class nflow()
```

<a id="nets.normflows.nflow.__init__"></a>

#### \_\_init\_\_

```python
def __init__(dim=2, latent=16, batchsize: int = 1, dataset=None)
```

The function __init__ initializes the class NormalizingFlowModel with the parameters dim,

latent, batchsize, and datasetPath

**Arguments**:

- `dim`: The dimension of the data, defaults to 2 (optional)
- `latent`: The number of latent variables in the model, defaults to 16 (optional)
- `batchsize` (`int (optional)`): The number of samples to generate at a time, defaults to 1
- `datasetPath` (`str (optional)`): The path to the dataset, defaults to data/dataset.csv

<a id="nets.normflows.nflow.compile"></a>

#### compile

```python
def compile(optim: torch.optim = torch.optim.Adam,
            distribution: str = 'GaussianKDE',
            lr: float = 0.00015,
            bw: float = 0.1,
            wd=0.0015)
```

It takes in a dataset, a model, and a distribution, and returns a compiled model

**Arguments**:

- `optim` (`torch.optim`): the optimizer to use
- `distribution` (`str (optional)`): the type of distribution to use, defaults to GaussianKDE
- `lr` (`float`): learning rate
- `bw` (`float`): bandwidth for the KDE

<a id="nets.normflows.nflow.train"></a>

#### train

```python
def train(iters: int = 1000)
```

> We sample from a normal distribution, pass the samples through the model, and then calculate

the loss

**Arguments**:

- `iters` (`int (optional)`): number of iterations to train for, defaults to 1000

<a id="nets.normflows.nflow.performance"></a>

#### performance

```python
def performance()
```

The function takes the model and the scaled data as inputs, samples from the model, and then
prints the r2 score of the samples and the scaled data.

<a id="nets.design"></a>

# nets.design

<a id="nets.design.B_field_norm"></a>

#### B\_field\_norm

```python
def B_field_norm(Bmax: float, L: float, k: int = 16, plot=True) -> np.array
```

Returns vec B_z for MS config

**Arguments**:

- `Bmax` _any_ - maximum B in thruster
  L - channel length
  k - magnetic field profile number

<a id="nets.design.PUdesign"></a>

#### PUdesign

```python
def PUdesign(P: float, U: float) -> pd.DataFrame
```

Computes design via numerical model, uses fits from PINNs

**Arguments**:

- `P` _float_ - _description_
- `U` _float_ - _description_
  

**Returns**:

- `_type_` - _description_

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

<a id="nets.opti"></a>

# nets.opti

<a id="nets.opti.blackbox"></a>

# nets.opti.blackbox

<a id="nets.opti.blackbox.Hyper"></a>

## Hyper Objects

```python
class Hyper(SCI)
```

Hyper parameter tunning class. Allows to generate best NN architecture for task. Inputs are column indexes. idx[-1] is targeted value.

<a id="nets.opti.blackbox.Hyper.start_study"></a>

#### start\_study

```python
def start_study(n_trials: int = 100,
                neptune_project: str = None,
                neptune_api: str = None)
```

Starts study. Optionally provide your neptune repo and token for report generation.

**Arguments**:

- `n_trials` _int, optional_ - Number of iterations. Defaults to 100.
- `neptune_project` _str, optional_ - . Defaults to None.
  neptune_api (str, optional):. Defaults to None.
  

**Returns**:

- `dict` - quick report of results

