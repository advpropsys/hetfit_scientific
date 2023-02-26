<a id="nets.opti.blackbox"></a>

# :orange[Hyper Paramaters Optimization class]
## nets.opti.blackbox

<a id="nets.opti.blackbox.Hyper"></a>

### Hyper Objects

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

