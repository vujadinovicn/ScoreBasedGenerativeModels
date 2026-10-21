# Score-Based Generative Modeling

A (student) reproduction and extension of *"Generative Modeling by Estimating Gradients of the Data Distribution"* (Song & Ermon, 2019). This repository implements a noise-conditional score network (NCSN) with annealed Langevin dynamics in PyTorch, trained on a subset of Fashion-MNIST, and uses it to study the choice of variance schedule and generation in a PCA latent space.

---

## Table of Contents

- [Overview](#overview)
- [Report](#report)
- [Dependencies](#dependencies)
- [Code structure](#code-structure)
- [Training](#training)
- [Sampling](#sampling)
- [Generated examples](#generated-examples)
- [Reference](#reference)

---

## Overview

**Task.** Implementation of score-based generative models and generating Fashion-MNIST images with them.

**What we implemented.**
- **Reproduction.** Noise-conditional score network with a RefineNet (dilated, conditional instance norm) backbone, denoising score matching training over `L` noise levels, and annealed Langevin dynamics sampling.
- **Variance schedules.** Besides the original geometric schedule, linear, square-root, quadratic and sigmoid discrete schedules for the noise levels `σ₁ … σ_L` (see `noise_schedulers.py`).
- **Continuous variance (proof of concept).** A 2D mixture-of-Gaussians toy experiment where the score network is conditioned on a continuous `σ` sampled log-uniformly, allowing arbitrary schedules (and fewer steps) at sampling time.
- **Latent-space generation.** Score estimation and sampling in a PCA latent space (e.g. 64 components, 8×8 latent images), mapped back to pixels with the inverse PCA. Training and sampling are noticeably faster than in data space.

---

## Report

For a more detailed description of our work, you can read the report [here](docs/paper.pdf).

---

## Dependencies

You can either create a new conda environment using the provided `environment.yml` file, or install the required packages in an existing environment.

```bash
conda env create -f environment.yml -n gda
```

```bash
pip install -r requirements.txt
```

---

## Code structure

| File | Description |
|---|---|
| `ncsn_model.py` | Noise-conditional score network (`CondRefineNetDilated`) |
| `denoising_dataloader.py` | Dataset that loads (Fashion-)MNIST, optionally projects it with PCA, and builds noisy copies at every noise level |
| `noise_schedulers.py` | Discrete variance schedules (geometric, linear, quadratic, sqrt, sigmoid) and plotting helpers |
| `training_code.py` | Training script (data space or PCA latent space) |
| `scheduler_trainings.py` | Placeholder showing how training is parametrised by schedule type; the actual runs were done in a Colab notebook (TODO: merge into `training_code.py`) |
| `sampling.py` | Annealed Langevin sampling from the data-space model |
| `sampling_pca.py` | Annealed Langevin sampling in the PCA latent space, followed by inverse PCA |
| `docs/` | Project report |
| `assets/` | Generated sample images used in this README |

---

## Training

```bash
python training_code.py \
  --total-steps 10000 \
  --batch-size 64 \
  --allowed-classes 0 1 2 \
  --L 10 \
  --sigma-1 1.0 \
  --sigma-L 0.01 \
  --lr 0.001 \
  --ngf 64 \
  --save-prefix MYMODEL
```

You can pass the following arguments to allow training in the latent space of PCA:
- `--PCA-components`: Number of PCA components to use (must be a perfect square, e.g. 64).
- `--PCA-model-save-path`: Path to save the trained PCA model.

---

## Sampling

### From data space
```bash
python sampling.py \
    --batch-size 8 \
    --model-path author_code/mnist/SAVED_MODEL_5.pt \
    --show
```

### From latent space
```bash
python sampling_pca.py \
    --batch-size 8 \
    --model-path PCA/SAVED_MODEL_5_fmnist.pt \
    --pca-path author_code/PCA/PCA_64_012.joblib \
    --show
```

In both sampling methods, you can also save the generated samples using the `--out` argument which requires a path to save the samples.

Trained checkpoints are not tracked in git, so train a model first or point `--model-path` to your own checkpoint.

---

## Generated examples

**Raw data generation**
![Raw data generation](assets/data_gen.png)

**PCA latent space generation**
![PCA latent space generation](assets/pca_gen.png)

---

## Reference

> Yang Song and Stefano Ermon.
> **"Generative Modeling by Estimating Gradients of the Data Distribution."**
> *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
