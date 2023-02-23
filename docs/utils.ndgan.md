<!-- markdownlint-disable -->

<a href="../utils/ndgan.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `utils.ndgan`






---

<a href="../utils/ndgan.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DCGAN`




<a href="../utils/ndgan.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(latent, data)
```








---

<a href="../utils/ndgan.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build_models`

```python
build_models()
```





---

<a href="../utils/ndgan.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `define_discriminator`

```python
define_discriminator(inputs=8)
```

function to return the compiled discriminator model 

---

<a href="../utils/ndgan.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `define_gan`

```python
define_gan(generator, discriminator)
```

define the combined generator and discriminator model 

---

<a href="../utils/ndgan.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `define_generator`

```python
define_generator(latent_dim, outputs=8)
```





---

<a href="../utils/ndgan.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate_fake_samples`

```python
generate_fake_samples(generator, latent_dim, n)
```





---

<a href="../utils/ndgan.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate_latent_points`

```python
generate_latent_points(latent_dim, n)
```

generate points in latent space as input for the generator 

---

<a href="../utils/ndgan.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(n)
```





---

<a href="../utils/ndgan.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `start_training`

```python
start_training()
```





---

<a href="../utils/ndgan.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `summarize_performance`

```python
summarize_performance(epoch, generator, discriminator, latent_dim, n=200)
```

evaluate the discriminator and plot real and fake samples 

---

<a href="../utils/ndgan.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train_gan`

```python
train_gan(
    g_model,
    d_model,
    gan_model,
    latent_dim,
    num_epochs=2500,
    num_eval=2500,
    batch_size=2
)
```

function to train gan model 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
