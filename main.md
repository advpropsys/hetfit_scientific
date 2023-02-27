# Table of Contents

- [Table of Contents](#table-of-contents)
- [main](#main)
- [PINN](#pinn)
- [PINN.pinns](#pinnpinns)
  - [PINNd\_p Objects](#pinnd_p-objects)
      - [forward](#forward)
  - [PINNhd\_ma Objects](#pinnhd_ma-objects)
  - [PINNT\_ma Objects](#pinnt_ma-objects)
- [utils](#utils)
- [utils.test](#utilstest)
- [utils.dataset\_loader](#utilsdataset_loader)
      - [get\_dataset](#get_dataset)
- [utils.ndgan](#utilsndgan)
  - [DCGAN Objects](#dcgan-objects)
      - [\_\_init\_\_](#__init__)
      - [define\_discriminator](#define_discriminator)
      - [define\_generator](#define_generator)
      - [build\_models](#build_models)
      - [generate\_latent\_points](#generate_latent_points)
      - [generate\_fake\_samples](#generate_fake_samples)
      - [define\_gan](#define_gan)
      - [summarize\_performance](#summarize_performance)
      - [train\_gan](#train_gan)
      - [start\_training](#start_training)
      - [predict](#predict)
- [utils.data\_augmentation](#utilsdata_augmentation)
  - [dataset Objects](#dataset-objects)
      - [\_\_init\_\_](#__init__-1)
      - [generate](#generate)
- [:orange\[nets\]](#orangenets)
- [nets.envs](#netsenvs)
  - [SCI Objects](#sci-objects)
      - [\_\_init\_\_](#__init__-2)
      - [feature\_gen](#feature_gen)
      - [feature\_importance](#feature_importance)
      - [data\_flow](#data_flow)
      - [init\_seed](#init_seed)
      - [train\_epoch](#train_epoch)
      - [compile](#compile)
      - [train](#train)
      - [save](#save)
      - [onnx\_export](#onnx_export)
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
      - [\_\_init\_\_](#__init__-3)
- [nets.design](#netsdesign)
      - [B\_field\_norm](#b_field_norm)
      - [PUdesign](#pudesign)
- [nets.deep\_dense](#netsdeep_dense)
  - [dmodel Objects](#dmodel-objects)
      - [\_\_init\_\_](#__init__-4)
- [nets.opti](#netsopti)
- [nets.opti.blackbox](#netsoptiblackbox)
  - [Hyper Objects](#hyper-objects)
      - [\_\_init\_\_](#__init__-5)
      - [define\_model](#define_model)
      - [objective](#objective)
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

<a id="PINN.pinns.PINNd_p.forward"></a>

#### forward

```python
def forward(x)
```

$P,U$ input, $d$ output

**Arguments**:

- `x` __type__ - _description_
  

**Returns**:

- `_type_` - _description_

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

<a id="utils.ndgan.DCGAN.__init__"></a>

#### \_\_init\_\_

```python
def __init__(latent, data)
```

The function takes in two arguments, the latent space dimension and the dataframe. It then sets

the latent space dimension, the dataframe, the number of inputs and outputs, and then builds the
models

**Arguments**:

- `latent`: The number of dimensions in the latent space
- `data`: This is the dataframe that contains the data that we want to generate

<a id="utils.ndgan.DCGAN.define_discriminator"></a>

#### define\_discriminator

```python
def define_discriminator(inputs=8)
```

The discriminator is a neural network that takes in a vector of length 8 and outputs a single

value between 0 and 1

**Arguments**:

- `inputs`: number of features in the dataset, defaults to 8 (optional)

**Returns**:

The model is being returned.

<a id="utils.ndgan.DCGAN.define_generator"></a>

#### define\_generator

```python
def define_generator(latent_dim, outputs=8)
```

The function takes in a latent dimension and outputs and returns a model with two hidden layers

and an output layer

**Arguments**:

- `latent_dim`: The dimension of the latent space, or the space that the generator will map
to
- `outputs`: the number of outputs of the generator, defaults to 8 (optional)

**Returns**:

The model is being returned.

<a id="utils.ndgan.DCGAN.build_models"></a>

#### build\_models

```python
def build_models()
```

The function returns the generator and discriminator models

**Returns**:

The generator and discriminator models are being returned.

<a id="utils.ndgan.DCGAN.generate_latent_points"></a>

#### generate\_latent\_points

```python
def generate_latent_points(latent_dim, n)
```

> Generate random points in latent space as input for the generator

**Arguments**:

- `latent_dim`: the dimension of the latent space, which is the input to the generator
- `n`: number of images to generate

**Returns**:

A numpy array of random numbers.

<a id="utils.ndgan.DCGAN.generate_fake_samples"></a>

#### generate\_fake\_samples

```python
def generate_fake_samples(generator, latent_dim, n)
```

It generates a batch of fake samples with class labels

**Arguments**:

- `generator`: The generator model that we will train
- `latent_dim`: The dimension of the latent space, e.g. 100
- `n`: The number of samples to generate

**Returns**:

x is the generated images and y is the labels for the generated images.

<a id="utils.ndgan.DCGAN.define_gan"></a>

#### define\_gan

```python
def define_gan(generator, discriminator)
```

The function takes in a generator and a discriminator, sets the discriminator to be untrainable,

and then adds the generator and discriminator to a sequential model. The sequential model is then compiled with an optimizer and a loss function. 

The optimizer is adam, which is a type of gradient descent algorithm. 

Loss function is binary crossentropy, which is a loss function that is used for binary
classification problems. 


The function then returns the GAN.

**Arguments**:

- `generator`: The generator model
- `discriminator`: The discriminator model that takes in a dataset and outputs a single value
representing fake/real

**Returns**:

The model is being returned.

<a id="utils.ndgan.DCGAN.summarize_performance"></a>

#### summarize\_performance

```python
def summarize_performance(epoch, generator, discriminator, latent_dim, n=200)
```

> This function evaluates the discriminator on real and fake data, and plots the real and fake

data

**Arguments**:

- `epoch`: the number of epochs to train for
- `generator`: the generator model
- `discriminator`: the discriminator model
- `latent_dim`: The dimension of the latent space
- `n`: number of samples to generate, defaults to 200 (optional)

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

**Arguments**:

- `g_model`: the generator model
- `d_model`: The discriminator model
- `gan_model`: The GAN model, which is the generator model combined with the discriminator
model
- `latent_dim`: The dimension of the latent space. This is the number of random numbers that
the generator model will take as input
- `num_epochs`: The number of epochs to train for, defaults to 2500 (optional)
- `num_eval`: number of epochs to run before evaluating the model, defaults to 2500
(optional)
- `batch_size`: The number of samples to use for each gradient update, defaults to 2
(optional)

<a id="utils.ndgan.DCGAN.start_training"></a>

#### start\_training

```python
def start_training()
```

The function takes the generator, discriminator, and gan models, and the latent vector as
arguments, and then calls the train_gan function.

<a id="utils.ndgan.DCGAN.predict"></a>

#### predict

```python
def predict(n)
```

It takes the generator model and the latent space as input and returns a batch of fake samples

**Arguments**:

- `n`: the number of samples to generate

**Returns**:

the generated fake samples.

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


**Arguments**:

- `number_samples` _int_ - number of samples to be genarated
- `name` _str_ - name of dataset
- `source` _str_ - source file
- `boundary_conditions` _list_ - y1,y2,x1,x2

<a id="utils.data_augmentation.dataset.generate"></a>

#### generate

```python
def generate()
```

The function takes in a dataframe, normalizes it, and then trains a DCGAN on it. 

The DCGAN is a type of generative adversarial network (GAN) that is used to generate new data. 

The DCGAN is trained on the normalized dataframe, and then the DCGAN is used to generate new
data. 

The new data is then concatenated with the original dataframe, and the new dataframe is saved as
a pickle file. 

The new dataframe is then returned.

**Returns**:

The dataframe is being returned.

<a id="nets"></a>

# :orange[nets]

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
             boundary_conditions: list = None,
             batch_size: int = 20)
```



**Arguments**:

- `hidden_dim` _int, optional_ - Max demension of hidden linear layer. Defaults to 200. Should be >80 in not 1d case
- `dropout` _bool, optional_ - LEGACY, don't use. Defaults to True.
- `epochs` _int, optional_ - Optionally specify epochs here, but better in train. Defaults to 10.
- `dataset` _str, optional_ - dataset to be selected from ./data. Defaults to 'test.pkl'. If name not exists, code will generate new dataset with upcoming parameters.
- `sample_size` _int, optional_ - Samples to be generated (note: BEFORE applying boundary conditions). Defaults to 1000.
- `source` _str, optional_ - Source from which data will be generated. Better to not change. Defaults to 'dataset.csv'.
- `boundary_conditions` _list, optional_ - If sepcified, whole dataset will be cut rectangulary. Input list is [ymin,ymax,xmin,xmax] type. Defaults to None.
- `batch_size` _int, optional_ - Batch size for training.

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

Initializes seed for torch - optional

<a id="nets.envs.SCI.train_epoch"></a>

#### train\_epoch

```python
def train_epoch(X, model, loss_function, optim)
```

Inner function of class - don't use.

We iterate through the data, calculate the loss, backpropagate, and update the weights

**Arguments**:

- `X`: the training data
- `model`: the model we're training
- `loss_function`: the loss function to use
- `optim`: the optimizer, which is the algorithm that will update the weights of the model

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
- `optim` - torch Optimizer. Default AdamW
- `loss` - torch Loss function (nn). Defaults to L1Loss

<a id="nets.envs.SCI.train"></a>

#### train

```python
def train(epochs: int = 10) -> None
```

Train model
- If sklearn instance uses .fit()

- epochs (int,optional)

<a id="nets.envs.SCI.save"></a>

#### save

```python
def save(name: str = 'model.pt') -> None
```

> This function saves the model to a file

**Arguments**:

- `name` (`str (optional)`): The name of the file to save the model to, defaults to model.pt

<a id="nets.envs.SCI.onnx_export"></a>

#### onnx\_export

```python
def onnx_export(path: str = './models/model.onnx')
```

> We are exporting the model to the ONNX format, using the input data and the model itself

**Arguments**:

- `path` (`str (optional)`): The path to save the model to, defaults to ./models/model.onnx

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

> If the input and output dimensions are the same, plot the input and output as a scatter plot.
If the input and output dimensions are different, plot the first dimension of the input and
output as a scatter plot

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

The Net class inherits from the nn.Module class, which has a number of attributes and methods (such
as .parameters() and .zero_grad()) which we will be using. You can read more about the nn.Module
class [here](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)

<a id="nets.dense.Net.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_dim: int = 2, hidden_dim: int = 200)
```

We create a neural network with two hidden layers, each with **hidden_dim** neurons, and a ReLU activation

function. The output layer has one neuron and no activation function

**Arguments**:

- `input_dim` (`int (optional)`): The dimension of the input, defaults to 2
- `hidden_dim` (`int (optional)`): The number of neurons in the hidden layer, defaults to 200

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

<a id="nets.deep_dense.dmodel.__init__"></a>

#### \_\_init\_\_

```python
def __init__(in_features=1, hidden_features=200, out_features=1)
```

We're creating a neural network with 4 layers, each with 200 neurons. The first layer takes in the input, the second layer takes in the output of the first layer, the third layer takes in the
output of the second layer, and the fourth layer takes in the output of the third layer

**Arguments**:

- `in_features`: The number of input features, defaults to 1 (optional)
- `hidden_features`: the number of neurons in the hidden layers, defaults to 200 (optional)
- `out_features`: The number of classes for classification (1 for regression), defaults to 1
(optional)

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
Based on OPTUNA algorithms it is very fast and reliable. Outputs are NN parameters in json. Optionally full report for every trial is available at the neptune.ai

<a id="nets.opti.blackbox.Hyper.__init__"></a>

#### \_\_init\_\_

```python
def __init__(idx: tuple = (1, 3, 7), *args, **kwargs)
```

The function __init__() is a constructor that initializes the class Hyper

**Arguments**:

- `idx` (`tuple`): tuple of integers, the indices of the data to be loaded

<a id="nets.opti.blackbox.Hyper.define_model"></a>

#### define\_model

```python
def define_model(trial)
```

We define a function that takes in a trial object and returns a neural network with the number

of layers, hidden units and activation functions defined by the trial object.

**Arguments**:

- `trial`: This is an object that contains the information about the current trial

**Returns**:

A sequential model with the number of layers, hidden units and activation functions
defined by the trial.

<a id="nets.opti.blackbox.Hyper.objective"></a>

#### objective

```python
def objective(trial)
```

We define a model, an optimizer, and a loss function. We then train the model for a number of

epochs, and report the loss at the end of each epoch

*"optimizer": ["Adam", "RMSprop", "SGD" 'AdamW','Adamax','Adagrad']*
*"lr" $\in$ [1e-7,1e-3], log=True*

**Arguments**:

- `trial`: The trial object that is passed to the objective function

**Returns**:

The accuracy of the model.

<a id="nets.opti.blackbox.Hyper.start_study"></a>

#### start\_study

```python
def start_study(n_trials: int = 100,
                neptune_project: str = None,
                neptune_api: str = None)
```

It takes a number of trials, a neptune project name and a neptune api token as input and runs

the objective function on the number of trials specified. If the neptune project and api token
are provided, it logs the results to neptune

**Arguments**:

- `n_trials` (`int (optional)`): The number of trials to run, defaults to 100
- `neptune_project` (`str`): the name of the neptune project you want to log to
- `neptune_api` (`str`): your neptune api key

