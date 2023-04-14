# A deep-learning-based denoiser: to reconstruct ground displacement from the original MPIC, which is applicable if at least nine optical image correlation maps are available.

# Installation
## Basic
- `Python` >= 3.8
## Modules

- `h5py`
- `numpy`
- `matplotlib` <= 3.5.1
- `rasterio`
- `tensorflow-gpu` >= 2.5.0

```shell
pip install -r requirements.txt
```


# Prediction
```shell
python 3DCNN_denoiser_MPIC.py
```
