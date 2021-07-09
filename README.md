## Official implementation of [Neural Distributed Source Coding](https://arxiv.org/abs/2106.02797)


## Setup
### Environment
* `Ubuntu Bionic 18.04.5 LTS`
* `Python 3.8.8, OpenMPI-2.1.1, NCCL-2.8.4`
* `PyTorch 1.8.0+cu111, Tensorflow 2.4.1, Horovod 0.21.3`
* `CUDA 11.0, cuDNN 8.0.5, cudatoolkit 11.0.221`

### Installation
We use [Anaconda](https://www.anaconda.com/products/individual) for managing Python environment.
```shell
conda env create --file environment.yml
conda activate neural_dsc
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
HOROVOD_GPU_OPERATIONS=NCCL pip install --upgrade --no-cache-dir "horovod[pytorch]"
```

----

## Experiment 1: Distributed Image Compression

### Activate environment
```shell
conda activate neural_dsc
```

### Prepare data (CelebA-HQ 256x256)
Download `celeba-tfr.tar` inside `data/` directory, then run the following command:
```shell
python run_top.py prep celebahq256
```

### Train VQ-VAE
Repeat the following with different `--codebook_bits` argument to control the total rate.
```shell
# Joint VQ-VAE
horovodrun -n 2 python -O run_top.py train --dataset celebahq256 --arch vqvae_top_8x --ch_latent 1 \
  --root_dir checkpoints/celeba256_vqvae_joint_4bit --dec_si True --enc_si True  --codebook_bits 4

# Distributed VQ-VAE
horovodrun -n 2 python -O run_top.py train --dataset celebahq256 --arch vqvae_top_8x --ch_latent 1 \
  --root_dir checkpoints/celeba256_vqvae_dist_4bit --dec_si True --enc_si False --codebook_bits 4

# Separate VQ-VAE
horovodrun -n 2 python -O run_top.py train --dataset celebahq256 --arch vqvae_top_8x --ch_latent 1 \
  --root_dir checkpoints/celeba256_vqvae_separate_4bit --dec_si False --enc_si False --codebook_bits 4
```

### Evaluate VQ-VAE
```shell
horovodrun -n 2 python run_top.py eval --batch_size 250 \
  checkpoints/celebahq256_vqvae_{joint,dist,separate}_4bit/ckpt_ep=020_step=0016880.pt
```

### Plot rate-distortion curves from eval results
All generated plots will be stored in the folder `paper/`.
```shell
python plot_rd_curves.py
```

----

## Experiment 2: Distributed SGD

### Activate environment
```shell
conda activate neural_dsc
```

### Prepare data
```shell
# Following command may take a while to finish due to slow download speed.
python run_mnist_grad.py prep mnist
```

### Gather gradients
```shell
python -O run_mnist_grad.py gather_gradients --out_dir checkpoints/mnist_grad_data
```

### Train VQ-VAE
```shell
# Joint VQ-VAE
python -O run_mnist_grad.py train_vqvae --grad_dump checkpoints/mnist_grad_data/grads.pt --d_latent 40 --codebook_bits 8 \
  --enc_si True  --dec_si True  --root_dir checkpoints/mnist_grad_vqvae_joint_40d_8bits

# Distributed VQ-VAE
python -O run_mnist_grad.py train_vqvae --grad_dump checkpoints/mnist_grad_data/grads.pt --d_latent 40 --codebook_bits 8 \
  --enc_si False --dec_si True  --root_dir checkpoints/mnist_grad_vqvae_dist_40d_8bits

# Separate VQ-VAE
python -O run_mnist_grad.py train_vqvae --grad_dump checkpoints/mnist_grad_data/grads.pt --d_latent 40 --codebook_bits 8 \
  --enc_si False --dec_si False --root_dir checkpoints/mnist_grad_vqvae_separate_40d_8bits
```

### Evaluate
```shell
for seed in $(seq 1 20); do
    python run_mnist_grad.py eval checkpoints/mnist_grad_vqvae_{joint,dist,separate}_40d_8bits/ckpt_ep=500_step=0391000.pt --seed $seed;
done
```

### Plot
```shell
python run_mnist_grad.py plot checkpoints/mnist_grad_vqvae_{joint,dist,separate}_40d_8bits/ckpt_ep=500_step=0391000.pt \
  --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --out_dir paper --labels Joint,Distributed,Separate

```
