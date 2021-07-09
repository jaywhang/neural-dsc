import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import itertools
import json
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import List

# For plots
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
import seaborn as sns; sns.set(context='paper', style='whitegrid', font_scale=2.0, font='Times New Roman')

import fire
import horovod.torch as hvd
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data.distributed as torch_dist
from torch.utils.data import TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm, trange

from code.utils import HParams, Logger, check_or_mkdir, seed_everything, get_param_count
from code.data import prepare_dataset, load_dataset
from code.arch import MnistClassifierCnn, VqvaeMnistGradTc


_MARKERS = ['o', '^', 's', 'x', '+']


def pp(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs, flush=True)


def plot_with_ci(x: torch.Tensor, y: torch.Tensor, ax: plt.Axes, alpha: float = 0.3, smooth=True, fill=True, **kwargs):
    assert len(x) == len(y) and x.ndim == 1 and y.ndim == 2
    x, y = x.cpu().numpy(), y.cpu().numpy()
    means = y.mean(axis=1)
    stderr = y.std(axis=1) / np.sqrt(y.shape[1])
    # stderr = y.std(axis=1)

    # Filter anomalies
    x = x[means <= 5]
    stderr = stderr[means <= 5]
    means = means[means <= 5]

    if smooth:
        w = 50
        means[w-1:] = np.convolve(means, np.ones(w), 'valid') / w
    ax.plot(x, means, **kwargs)

    if fill:
        ax.fill_between(x, means-stderr, means+stderr, alpha=alpha)

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)


@torch.no_grad()
def eval_cnn(model, dataset_):
    model.eval()
    xs = dataset_.data.cuda().float() / 255.  - 0.13066047430038452
    ys = dataset_.label.cuda()
    losses, preds = [], []
    assert len(dataset_) in (60000, 10000)
    for i in range(len(dataset_) // 500):
        x = xs[i*500 : (i+1)*500]
        y = ys[i*500 : (i+1)*500]
        yp = model(x)
        losses.append(F.cross_entropy(yp, y, reduction='none'))
        preds.append(y == yp.argmax(dim=1))
    losses = torch.cat(losses)
    preds = torch.cat(preds).float()
    assert losses.shape == preds.shape == (len(dataset_),)
    model.train()
    return losses.mean(), preds.mean()


def test_cnn(*,
             out_dir: str,
             seed: int = 1337,
             num_runs: int = 2,
             num_steps: int = 2500,
             learning_rate: float = 1.0,
             lr_schedule: List[int] = [50, 200, 350],
             lr_decay: float = 0.1,
             batch_size: int = 3000,
             data_root: str = None):
    assert batch_size <= 15000 and 15000 % batch_size == 0 and num_steps % (15000 // batch_size) == 0
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(seed)

    # Load data
    mnist_dataset = load_dataset('mnist', split='train', data_root=data_root)
    perm_idx = np.random.default_rng(0).choice(len(mnist_dataset), len(mnist_dataset), replace=False)
    raw_data, raw_labels = mnist_dataset.data[perm_idx], mnist_dataset.label[perm_idx]
    raw_data = raw_data.float() / 255. - 0.13066047430038452
    raw_data, raw_labels = raw_data.cuda(), raw_labels.cuda()
    data1, labels1 = raw_data[:15000], raw_labels[:15000]
    data2, labels2 = raw_data[15000:30000], raw_labels[15000:30000]
    assert data1.shape == data2.shape == (15000, 1, 28, 28) and labels1.shape == labels2.shape == (15000,)
    dataset_val = load_dataset('mnist', split='val', data_root=data_root)

    stats = {
        'train_losses': [],
        'val_losses': [],
        'val_accs': [],
    }

    def _lr_scheduler(ep: int, cur_lr: float, decay=lr_decay, schedule=lr_schedule):
        if schedule is None or ep not in schedule:
            return cur_lr
        new_lr = cur_lr * decay
        print(f'LR decay at epoch {ep}: {cur_lr:.4f} -> {new_lr:.4f}')
        return new_lr

    # Train and gather gradients
    for run_id in range(num_runs):
        lr = learning_rate
        epoch = 0

        # Create model
        model = MnistClassifierCnn().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        step = 0
        run_loss, run_val_loss, run_val_acc = [], [], []
        pbar = tqdm(total=num_steps, desc=f'Run [{run_id+1} / {num_runs}]', leave=False)

        # Shuffle data
        idx = np.random.choice(len(data1), len(data1), replace=False)
        d1, d2, l1, l2 = data1[idx], data2[idx], labels1[idx], labels2[idx]
        while step < num_steps:
            epoch += 1
            for i in range(len(data1) // batch_size):
                x1 = d1[i*batch_size : (i+1)*batch_size]
                x2 = d2[i*batch_size : (i+1)*batch_size]
                y1 = l1[i*batch_size : (i+1)*batch_size]
                y2 = l2[i*batch_size : (i+1)*batch_size]
                loss1 = F.cross_entropy(model(x1), y1)
                loss2 = F.cross_entropy(model(x2), y2)
                loss = (loss1 + loss2) / 2
                run_loss.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                step += 1
                pbar.update(1)

            lr = _lr_scheduler(epoch, lr)
            for g in opt.param_groups:
                g['lr'] = lr

            val_loss, val_pred = eval_cnn(model, dataset_val)
            run_val_loss.append(val_loss.item())
            run_val_acc.append(val_pred.item())

        pbar.close()
        del model
        stats['train_losses'].append(run_loss)
        stats['val_losses'].append(run_val_loss)
        stats['val_accs'].append(run_val_acc)
        print(f'Run [{run_id+1} / {num_runs}] train_loss {run_loss[-1]:.4f} val_loss {run_val_loss[-1]:.4f} val_acc {run_val_acc[-1]:.4f} ')

    assert len(stats['train_losses']) == len(stats['val_losses']) == len(stats['val_accs']) == num_runs
    stats['train_losses'] = torch.Tensor(stats['train_losses'])
    stats['val_losses'] = torch.Tensor(stats['val_losses'])
    stats['val_accs'] = torch.Tensor(stats['val_accs'])
    assert stats['train_losses'].shape == (num_runs, num_steps)
    assert stats['val_losses'].shape == stats['val_accs'].shape == (num_runs, num_steps // (15000 // batch_size))
    torch.save(stats, os.path.join(out_dir, 'stats.pt'))


def gather_gradients(*,
                     out_dir: str,
                     seed: int = 1337,
                     num_runs: int = 120,
                     samples_per_run: int = 1000,
                     num_steps: int = 1000,
                     learning_rate: float = 0.01,
                     lr_schedule: List[int] = None,
                     lr_decay: float = None,
                     batch_size: int = 300,
                     data_root: str = None):
    assert samples_per_run <= num_steps and batch_size <= 15000
    assert 15000 % batch_size == 0 and num_steps % (15000 // batch_size) == 0
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(seed)

    # Load data
    mnist_dataset = load_dataset('mnist', split='train', data_root=data_root)
    perm_idx = np.random.default_rng(0).choice(len(mnist_dataset), len(mnist_dataset), replace=False)
    raw_data, raw_labels = mnist_dataset.data[perm_idx].cuda(), mnist_dataset.label[perm_idx].cuda()
    raw_data = raw_data.float() / 255. - 0.13066047430038452
    data1, labels1 = raw_data[:15000], raw_labels[:15000]
    data2, labels2 = raw_data[15000:30000], raw_labels[15000:30000]
    assert data1.shape == data2.shape == (15000, 1, 28, 28) \
        and labels1.shape == labels2.shape == (15000,)
    dataset_tr = TensorDataset(data1, labels1, data2, labels2)
    dataset_val = load_dataset('mnist', split='val', data_root=data_root)

    args = dict(
        out_dir         = out_dir,
        seed            = seed,
        num_runs        = num_runs,
        samples_per_run = samples_per_run,
        num_steps       = num_steps,
        learning_rate   = learning_rate,
        lr_schedule     = lr_schedule,
        lr_decay        = lr_decay,
        batch_size      = batch_size,
        data_root       = data_root,
    )
    with open(os.path.join(out_dir, 'grad_args.json'), 'w') as f:
        json.dump(args, f, indent=2)

    grads = []
    stats = {
        'args': args,
        'train_losses': [],
        'val_losses': [],
        'val_accs': [],
    }

    def _lr_scheduler(ep: int, cur_lr: float, decay=lr_decay, schedule=lr_schedule):
        if schedule is None or ep not in schedule:
            return cur_lr
        new_lr = cur_lr * decay
        return new_lr

    step_dist = np.ones(num_steps)
    step_dist = step_dist / step_dist.sum()
    assert step_dist.shape == (num_steps,)

    # Train and gather gradients
    for run_id in range(num_runs):
        lr = learning_rate
        epoch = 0

        # Create model
        model = MnistClassifierCnn().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        params = [p for _, p in sorted(model.named_parameters(), key=lambda x: x[0])]
        target_steps = set(np.random.choice(num_steps, samples_per_run, replace=False, p=step_dist))
        step = 0
        assert len(target_steps) == samples_per_run
        run_loss, run_val_loss, run_val_acc = [], [], []
        pbar = tqdm(total=num_steps, desc=f'Run [{run_id+1} / {num_runs}]', leave=False)
        while step < num_steps:
            idx = np.random.choice(len(data1), len(data1), replace=False)
            d1, d2 = data1[idx], data2[idx]
            l1, l2 = labels1[idx], labels2[idx]
            epoch += 1
            for i in range(15000 // batch_size):
                x1 = d1[i*batch_size : (i+1)*batch_size]
                x2 = d2[i*batch_size : (i+1)*batch_size]
                y1 = l1[i*batch_size : (i+1)*batch_size]
                y2 = l2[i*batch_size : (i+1)*batch_size]

                # Compute gradients
                loss1 = F.cross_entropy(model(x1), y1)
                loss2 = F.cross_entropy(model(x2), y2)
                run_loss.append((loss1 + loss2).item() / 2)
                grad1 = torch.autograd.grad(loss1, params, retain_graph=True)
                grad2 = torch.autograd.grad(loss2, params, retain_graph=True)
                if step in target_steps:
                    grads.append((
                        step,
                        torch.cat([g.flatten() for g in grad1], dim=0).cpu(),
                        torch.cat([g.flatten() for g in grad2], dim=0).cpu(),
                    ))

                # Take gradient step
                optimizer.zero_grad()
                for p, g1, g2 in zip(params, grad1, grad2):
                    assert p.shape == g1.shape == g2.shape
                    p.grad = (g1 + g2) / 2.
                optimizer.step()

                step += 1
                pbar.update(1)

            lr = _lr_scheduler(epoch, lr)
            val_loss, val_pred = eval_cnn(model, dataset_val)
            run_val_loss.append(val_loss.item())
            run_val_acc.append(val_pred.item())

        pbar.close()
        del model
        stats['train_losses'].append(run_loss)
        stats['val_losses'].append(run_val_loss)
        stats['val_accs'].append(run_val_acc)
        print(f'Run [{run_id+1} / {num_runs}] train_loss {run_loss[-1]:.4f} val_loss {run_val_loss[-1]:.4f} val_acc {run_val_acc[-1]:.4f} ')

    assert len(stats['train_losses']) == len(stats['val_losses']) == len(stats['val_accs']) == num_runs
    stats['train_losses'] = torch.Tensor(stats['train_losses'])
    stats['val_losses'] = torch.Tensor(stats['val_losses'])
    stats['val_accs'] = torch.Tensor(stats['val_accs'])
    assert stats['train_losses'].shape == (num_runs, num_steps)
    assert stats['val_losses'].shape == stats['val_accs'].shape == (num_runs, num_steps // (15000 // batch_size))
    torch.save(stats, os.path.join(out_dir, 'stats.pt'))

    # Save gathered gradients
    steps = torch.Tensor([t for t, _, _ in grads]).long()
    g1 = torch.stack([g for _, g, _ in grads])
    g2 = torch.stack([g for _, _, g in grads])
    assert g1.shape == g2.shape ==(num_runs * samples_per_run, 412)
    torch.save((steps, g1, g2), os.path.join(out_dir, 'grads.pt'))
    print(f'Stored gradients! Shape: {steps.shape}, {g1.shape}, {g2.shape}')


def train_vqvae(*,
                root_dir: str,
                grad_dump: str,
                seed: int = 1234,
                test_run: bool = False,

                # Model hyperparameters
                codebook_bits: int = 8,
                d_latent: int = 40,
                enc_si: bool,
                dec_si: bool,

                # Override default model hyperparameters
                **kwargs):
    # Argument validation
    args = {
        'root_dir': root_dir,
        'seed': seed,

        'codebook_bits': codebook_bits,
        'd_latent': d_latent,
        'enc_si': enc_si,
        'dec_si': dec_si,
    }
    for k, v in kwargs.items():
        assert k not in args
        args[k] = v

    # Horovod setup
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # Create output directory or load checkpoint
    start_epoch = 1
    total_steps = 0
    if hvd.rank() == 0:
        check_or_mkdir(root_dir)
        pp(f'Training in directory {root_dir}')

    # Load data 
    NUM_TRAIN, NUM_VAL = 100000, 20000
    timesteps, data1, data2 = torch.load(grad_dump)
    assert len(timesteps) == len(data1) == len(data2) == NUM_TRAIN + NUM_VAL
    idx = np.random.default_rng(seed).choice(len(timesteps), len(timesteps), replace=False)
    data1, data2 = data1[idx], data2[idx]
    dataset_tr = TensorDataset(timesteps[:NUM_TRAIN], data1[:NUM_TRAIN], data2[:NUM_TRAIN])
    dataset_val = TensorDataset(timesteps[NUM_TRAIN:], data1[NUM_TRAIN:], data2[NUM_TRAIN:])
    assert len(dataset_tr) == NUM_TRAIN and len(dataset_val) == NUM_VAL

    # Create model
    seed_everything(seed)
    model = VqvaeMnistGradTc(d_latent=d_latent, enc_si=enc_si, dec_si=dec_si, codebook_bits=codebook_bits, **kwargs)
    model.cuda()
    hp = model.hp

    loader_kwargs = {'num_workers': 6, 'pin_memory': True}
    if (loader_kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        loader_kwargs['multiprocessing_context'] = 'forkserver'

    local_batch_size = hp.batch_size // hvd.size()
    sampler_tr = torch_dist.DistributedSampler(dataset_tr, num_replicas=hvd.size(), rank=hvd.rank(), seed=seed)
    loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=local_batch_size, sampler=sampler_tr, **loader_kwargs)

    # Create optimizer
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hp.learning_rate)
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Save hparams and args
    if hvd.rank() == 0:
        logger = Logger(root_dir)
        hp.save(os.path.join(root_dir, 'hparams.json'))
        with open(os.path.join(root_dir, 'train_args.json'), 'w') as f:
            json.dump(args, f, indent=2)

    # Print some info
    params_grad = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    params_all = sum([np.prod(p.shape) for p in model.parameters()])
    pp(f'  >>> Trainable / Total params: {params_grad} / {params_all}')
    pp(f'  >>> Horovod local_size / size = {hvd.local_size()} / {hvd.size()}')
    pp(f'  >>> Per-GPU / Total batch size = {local_batch_size} / {hp.batch_size}')
    pp('Starting training!\n')

    start_time = time.time()
    stats = SimpleNamespace(
        loss                = [],
        loss_mse            = [],
        loss_commit         = [],
        codebook_use        = [],

        steps_per_sec       = [],
        total_time          = [],
        epoch               = [],

        train_mse           = [],
        val_mse             = [],
    )


    # Full train/validation run
    @torch.no_grad()
    def full_eval(dset):
        model.eval()
        sampler = torch_dist.DistributedSampler(dset, num_replicas=hvd.size(), rank=hvd.rank())
        loader = torch.utils.data.DataLoader(dset, batch_size=local_batch_size*4, sampler=sampler, drop_last=False, **loader_kwargs)
        sampler.set_epoch(0)
        mse_list = []
        for t, x1, x2 in loader:
            t = t.to('cuda', non_blocking=True)
            x1 = x1.to('cuda', non_blocking=True)
            x2 = x2.to('cuda', non_blocking=True)
            x1_rec, _, _, _, _ = model(x1, t, c=x2)

            loss_mse = (x1_rec - x1).pow(2).sum(dim=1)
            loss_mse = hvd.allgather(loss_mse)
            mse_list.append(loss_mse.cpu())

        model.train()
        mse_list = torch.cat(mse_list)
        assert mse_list.shape == (len(dset),)
        return mse_list

    local_steps = 0
    for epoch in (range(start_epoch, hp.max_epoch+1) if hp.max_epoch > 0 else itertools.count(start_epoch)):
        sampler_tr.set_epoch(epoch)
        for batch_idx, (t, x1, x2) in enumerate(loader_tr):
            if test_run and local_steps > 10:
                break
            total_steps += 1
            local_steps += 1
            t = t.to('cuda', non_blocking=True)
            x1 = x1.to('cuda', non_blocking=True)
            x2 = x2.to('cuda', non_blocking=True)

            x1_rec, z_q, emb, z_e, idx = model(x1, t, c=x2)
            loss_mse = (x1_rec - x1).pow(2).sum(-1).mean()
            loss_commit = (z_q.detach() - z_e).pow(2).view(len(t) * model.hp.d_latent, model.hp.embedding_dim).sum(-1).mean() * hp.beta
            loss = loss_mse + loss_commit
            unique_idx = idx.unique()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Monitoring
            total_time = time.time() - start_time
            stats.loss.append(loss.item())
            stats.loss_mse.append(loss_mse.item())
            stats.loss_commit.append(loss_commit.item())
            stats.codebook_use.append(len(unique_idx.unique()))
            stats.steps_per_sec.append(local_steps / total_time)
            stats.total_time.append(total_time)
            stats.epoch.append(epoch)

            if total_steps % hp.print_freq == 0 or batch_idx == len(loader_tr) - 1:
                # Horovod: use train_sampler to determine the number of examples in this worker's partition.
                pp(f'\rep {epoch:03d} step {batch_idx+1:07d}/{len(loader_tr)} '
                   f'total_steps {total_steps:06d} ',
                   f'loss {stats.loss[-1]:.4f} ',
                   f'loss_mse {stats.loss_mse[-1]:.4f} ',
                   f'loss_commit {stats.loss_commit[-1]:.4f} ',
                   f'codebook_use {stats.codebook_use[-1]:03d} ',
                   f'time {stats.total_time[-1]:.2f} sec ',
                   f'steps/sec {stats.steps_per_sec[-1]:.2f} ', end='')

            if hvd.rank() == 0 and total_steps % hp.log_freq == 0:
                logger.log_scalars({
                    'train/loss': stats.loss[-1],
                    'train/loss_mse': stats.loss_mse[-1],
                    'train/loss_commit': stats.loss_commit[-1],
                    'train/codebook_use': stats.codebook_use[-1],
                    'perf/epoch': stats.epoch[-1],
                    'perf/total_time': stats.total_time[-1],
                    'perf/steps_per_sec': stats.steps_per_sec[-1],
                }, total_steps)
        pp()

        if hvd.rank() == 0 and epoch % hp.eval_freq == 0:
            train_mse = full_eval(dataset_tr)
            val_mse = full_eval(dataset_val)
            stats.train_mse.append(train_mse.mean().item())
            stats.val_mse.append(val_mse.mean().item())
            train_mse = None
            val_mse = None
            print(f'  --> [Eval] train_mse = {stats.train_mse[-1]:.4f}   val_mse = {stats.val_mse[-1]:.4f}')

            logger.log_scalars({
                'eval/train_mse': stats.train_mse[-1],
                'eval/val_mse': stats.val_mse[-1],
            }, total_steps)

        if hvd.rank() == 0 and epoch % hp.ckpt_freq == 0:
            dump_dict = {
                'stats': vars(stats),
                'hparams': vars(hp),
                'args': args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),

                'epoch': epoch,
                'total_steps': total_steps,
            }
            torch.save(dump_dict, os.path.join(root_dir, f'ckpt_ep={epoch:03d}_step={total_steps:07d}.pt'))
            pp(f'[CHECKPOINT] Saved model at epoch {epoch}')

    pp('Training finished!')


def evaluate(*ckpts: str,
             overwrite: bool = False,
             seed: int = 123,
             learning_rate: float = 0.01,
             batch_size: int = 300,
             num_steps: int = 2500,
             compress: bool = True,
             init_ckpt: str = None,
             base_out_fn: str = None,
             lr_schedule: List[int] = None,
             lr_decay: float = None,
             data_root: str = None):
    assert all([os.path.basename(ckpt).startswith('ckpt_') and ckpt.endswith('.pt') for ckpt in ckpts]), \
        f'Checkpoint names must start with "ckpt_" and end with ".pt"'
    assert compress or base_out_fn is not None

    eval_args = dict(
        ckpts           = ckpts,
        overwrite       = overwrite,
        seed            = seed,
        learning_rate   = learning_rate,
        lr_schedule     = lr_schedule,
        lr_decay        = lr_decay,
        batch_size      = batch_size,
        num_steps       = num_steps,
        compress        = compress,
        init_ckpt       = init_ckpt,
        data_root       = data_root,
    )

    def _lr_scheduler(ep: int, cur_lr: float, decay=lr_decay, schedule=lr_schedule):
        if schedule is None or ep not in schedule:
            return cur_lr
        new_lr = cur_lr * decay
        print(f'LR decay at epoch {ep}: {cur_lr:.4f} -> {new_lr:.4f}')
        return new_lr

    # Load data
    mnist_dataset = load_dataset('mnist', split='train', data_root=data_root)
    perm_idx = np.random.default_rng(0).choice(len(mnist_dataset), len(mnist_dataset), replace=False)
    raw_data, raw_labels = mnist_dataset.data[perm_idx][30000:].cuda(), mnist_dataset.label[perm_idx][30000:].cuda()
    raw_data = raw_data.float() / 255. - 0.13066047430038452
    data1, labels1 = raw_data[:15000], raw_labels[:15000]
    data2, labels2 = raw_data[15000:], raw_labels[15000:]
    assert data1.shape == data2.shape == (15000, 1, 28, 28) and labels1.shape == labels2.shape == (15000,)
    dataset_train = load_dataset('mnist', split='train', data_root=data_root)
    dataset_val = load_dataset('mnist', split='val', data_root=data_root)

    results = []
    for ckpt in ckpts:
        ckpt_dir = os.path.dirname(os.path.expanduser(ckpt))
        out_fn = os.path.join(ckpt_dir, f'eval_seed={seed}_init={init_ckpt is not None}_{os.path.basename(ckpt)[5:]}')
        if os.path.exists(out_fn) and not overwrite and compress:
            print(f'Loading existing result_dict from {out_fn}...')
            result_dict = torch.load(out_fn)
        else:
            result_dict = eval_ckpt(ckpt, (data1, data2, labels1, labels2), dataset_train, dataset_val, batch_size, num_steps, learning_rate, seed, compress, init_ckpt, _lr_scheduler)
            if compress:
                torch.save(result_dict, out_fn)
                with open(os.path.join(ckpt_dir, 'eval_args.json'), 'w') as f:
                    json.dump(eval_args, f, indent=2)
                print('Compressed eval result:')
            else:
                torch.save(result_dict, base_out_fn)
                print('Uncompressed eval result:')

        print(f'  -> train_loss: {result_dict["train_losses"][-1]:.4f}')
        print(f'  -> epoch_train_loss : {result_dict["epoch_train_losses"][-1]:.4f}')
        print(f'  -> test_loss : {result_dict["test_losses"][-1]:.4f}')
        print(f'  -> test_acc  : {result_dict["test_accs"][-1]*100:.2f} %')
        print(f'  -> rate      : {result_dict["rate"]:.2f}')
        print(f'  -> avg MSE   : {result_dict["mses"].mean():.4f}')

        if not compress:
            break

    print('Evaluation finished!')


def eval_ckpt(ckpt, data_tuple, dataset_train, dataset_test, batch_size, num_steps, learning_rate, seed, compress: bool, init_ckpt: str, lr_scheduler):
    ckpt_path = os.path.expanduser(ckpt)
    ckpt_dir = os.path.dirname(ckpt_path)
    data1, data2, labels1, labels2 = data_tuple
    assert len(data1) == len(data2) == len(labels1) == len(labels2)
    assert os.path.isfile(ckpt_path) and os.path.isdir(ckpt_dir), f'ckpt_path: {ckpt_path} ckpt_dir: {ckpt_dir}'
    seed_everything(seed)

    # Load hparams and args from training, and save eval args
    hp = HParams.load(os.path.join(ckpt_dir, 'hparams.json'))
    with open(os.path.join(ckpt_dir, 'train_args.json'), 'r') as f:
        train_args = json.load(f)

    # Load/Create model
    vqvae = VqvaeMnistGradTc(**hp)
    dd = torch.load(os.path.expanduser(ckpt))
    vqvae.load_state_dict(dd['model_state_dict'])
    print(f'Loaded model weights from {ckpt}')
    vqvae.cuda().eval()
    model = MnistClassifierCnn().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if init_ckpt is not None:
        # state_dict = torch.load('checkpoints/mnist_grad_seed1/state_dict.pt')
        state_dict = torch.load(init_ckpt)
        model.load_state_dict(state_dict)
    params = [p for _, p in sorted(model.named_parameters(), key=lambda x: x[0])]

    # Train and gather gradients
    pbar = tqdm(total=num_steps)
    step = 0
    epoch = 0
    train_losses, mses = [], []
    epoch_train_losses, epoch_train_accs, test_losses, test_accs = [], [], [], []
    while step < num_steps:
        epoch += 1
        idx = np.random.choice(len(data1), len(data1), replace=False)
        d1, d2 = data1[idx], data2[idx]
        l1, l2 = labels1[idx], labels2[idx]
        for i in range(len(data1) // batch_size):
            step += 1
            x1 = d1[i*batch_size : (i+1)*batch_size]
            x2 = d2[i*batch_size : (i+1)*batch_size]
            y1 = l1[i*batch_size : (i+1)*batch_size]
            y2 = l2[i*batch_size : (i+1)*batch_size]

            # Compute gradients
            loss1 = F.cross_entropy(model(x1), y1)
            loss2 = F.cross_entropy(model(x2), y2)
            train_losses.append(((loss1 + loss2) / 2).item())
            grad1 = torch.autograd.grad(loss1, params, retain_graph=True)
            grad2 = torch.autograd.grad(loss2, params, retain_graph=True)

            # Compute g1, g2
            g1 = torch.autograd.grad(loss1, params)
            g2 = torch.autograd.grad(loss2, params)
            g1_flat = torch.cat([g.flatten() for g in g1], dim=0)
            g2_flat = torch.cat([g.flatten() for g in g2], dim=0)

            # Compress g1
            if compress:
                with torch.no_grad():
                    t = torch.ones(1).to(g1_flat).long() * (step-1)
                    g1_rec, _, _, _, _ = vqvae(g1_flat.view(1,-1), t, c=g2_flat.view(1,-1))
                g1_rec = g1_rec.view(-1)
                assert g1_rec.shape == g1_flat.shape == (412,)
            else:
                g1_rec = g1_flat
            mses.append((g1_rec - g1_flat).pow(2).sum().item())

            # Take gradient step
            optimizer.zero_grad()
            g_combined = (g1_rec + g2_flat) / 2.
            csum = 0
            for p in params:
                n = np.prod(p.shape).item()
                g = g_combined[csum:csum+n]
                p.grad = g.view(*p.shape)
                csum += n
            optimizer.step()

            pbar.update(1)
            if step >= num_steps:
                break

        epoch_train_loss, epoch_train_acc = eval_cnn(model, dataset_train)
        epoch_train_losses.append(epoch_train_loss.item())
        epoch_train_accs.append(epoch_train_acc.item())

        test_loss, test_acc = eval_cnn(model, dataset_test)
        test_losses.append(test_loss.item())
        test_accs.append(test_acc.item())
    mses = torch.Tensor(mses)
    pbar.close()

    assert len(test_losses) == len(test_accs) == epoch

    # Gather results
    total_rate = vqvae.hp.d_latent * vqvae.hp.codebook_bits
    result_dict = {
        'train_losses': train_losses,
        'epoch_train_losses': epoch_train_losses,
        'epoch_train_accs': epoch_train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'mses': mses,
        'rate': total_rate,
        'num_epochs': epoch,
    }

    return result_dict


def plot_vqvae(*ckpts: str,
               out_dir: str,
               labels: List[str]):
    assert os.path.isdir(out_dir)
    if isinstance(labels, str):
        labels = [labels]

    train_mses = dict()
    val_mses = dict()
    for ckpt, label in zip(ckpts, labels):
        assert os.path.exists(ckpt)
        dd = torch.load(ckpt, map_location=torch.device('cpu'))
        stats = dd['stats']
        train_mses[label] = stats['train_mse']
        val_mses[label] = stats['val_mse']

    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_title(f'Validation MSE')
    ax.set_xlim(20, 520)
    ax.set_ylim(0.18, 0.32)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_xticks([50, 150, 250, 350, 450])
    ax.set_xticklabels([50, 150, 250, 350, 450])
    ax.set_yticks([0.2, 0.25, 0.3])
    ax.set_yticklabels([0.2, 0.25, 0.3])
    for marker, label in zip(_MARKERS, labels):
        y = val_mses[label]
        x = (np.arange(len(y)) + 1) * 5
        x, y = x[::5], y[::5]
        ax.plot(x, y, label=label, marker=marker, markersize=6)
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'mnist_grad_vqvae_val_mse.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_title(f'Train MSE')
    ax.set_xlim(20, 520)
    ax.set_ylim(0.18, 0.32)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_xticks([50, 150, 250, 350, 450])
    ax.set_xticklabels([50, 150, 250, 350, 450])
    ax.set_yticks([0.2, 0.25, 0.3])
    ax.set_yticklabels([0.2, 0.25, 0.3])
    for marker, label in zip(_MARKERS, labels):
        y = train_mses[label]
        x = (np.arange(len(y)) + 1) * 5
        x, y = x[::5], y[::5]
        ax.plot(x, y, label=label, marker=marker, markersize=6)
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'mnist_grad_vqvae_train_mse.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot(*ckpts: str,
         labels: List[str],
         seeds: List[int],
         out_dir: str,
         max_epoch: int = 16,
         init: bool = False,
         smooth: bool = False):
    assert os.path.isdir(out_dir)

    if isinstance(labels, str):
        labels = [labels]
    if isinstance(seeds, int):
        seeds = [seeds]

    # Gather data from eval runs
    train_losses = {l: [] for l in labels}
    epoch_train_losses = {l: [] for l in labels}
    test_losses = {l: [] for l in labels}
    test_accs = {l: [] for l in labels}
    mses = {l: [] for l in labels}
    rate, num_epochs = None, None
    for ckpt, label in zip(ckpts, labels):
        for seed in seeds:
            ckpt_dir = os.path.dirname(os.path.expanduser(ckpt))
            out_fn = os.path.join(ckpt_dir, f'eval_seed={seed}_init={init}_{os.path.basename(ckpt)[5:]}')
            assert os.path.exists(out_fn), f'Log "{out_fn}" does not exist!'
            eval_result = torch.load(out_fn, map_location=torch.device('cpu'))

            if rate is None:
                rate = eval_result['rate']
                num_epochs = eval_result['num_epochs']
            else:
                assert rate == eval_result['rate'], f'Different rate provided'
                assert num_epochs == eval_result['num_epochs'], f'Different num_epochs provided'
            
            train_losses[label].append(eval_result['train_losses'])
            epoch_train_losses[label].append(eval_result['epoch_train_losses'])
            test_losses[label].append(eval_result['test_losses'])
            test_accs[label].append(eval_result['test_accs'])
            mses[label].append(eval_result['mses'])
        
        train_losses[label] = torch.Tensor(train_losses[label]).transpose(0, 1)
        mses[label] = torch.stack(mses[label]).transpose(0, 1)
        epoch_train_losses[label] = torch.Tensor(epoch_train_losses[label]).transpose(0, 1)
        test_losses[label] = torch.Tensor(test_losses[label]).transpose(0, 1)
        test_accs[label] = torch.Tensor(test_accs[label]).transpose(0, 1)
        assert epoch_train_losses[label].shape == test_losses[label].shape == test_accs[label].shape == (num_epochs, len(seeds))
        assert mses[label].shape == train_losses[label].shape and train_losses[label].shape[1] == len(seeds)

    max_epoch = max_epoch or len(test_accs[labels[0]])
    max_step = int(len(train_losses[labels[0]]) * max_epoch / len(test_accs[labels[0]]))

    # Print test loss/accuracy stats
    print(f'Loss/accuracy stats')
    for label in labels:
        print(f'{label}:')
        accs = test_accs[label][max_epoch-1] * 100
        losses = test_losses[label][max_epoch-1]
        epoch_train_loss = epoch_train_losses[label][max_epoch-1]
        print(f'   -> Test acc : {accs.mean():.2f} +/- {accs.std():.3f} %  (min: {accs.min().item():.2f}  max: {accs.max().item():.2f})')
        print(f'   -> Test loss: {losses.mean():.3f} +/- {losses.std():.3f}  (min: {losses.min().item():.3f}  max: {losses.max().item():.3f})')
        print(f'   -> Epoch train loss: {epoch_train_loss.mean():.3f} +/- {epoch_train_loss.std():.3f}  (min: {epoch_train_loss.min().item():.3f}  max: {epoch_train_loss.max().item():.3f})')

    # Load baseline results
    baseline_results = get_baseline_results()
    baseline_labels = list(baseline_results.keys())
    print(f'\nBASELINE RESULTS ')
    for label, res in baseline_results.items():
        print(f'{label}:')
        accs = res['test_accs'][max_epoch-1] * 100
        losses = res['epoch_train_losses'][max_epoch-1]
        print(f'   -> Train acc : {accs.mean():.2f} +/- {accs.std():.3f} %  (min: {accs.min().item():.2f}  max: {accs.max().item():.2f})')
        print(f'   -> Train loss: {losses.mean():.3f} +/- {losses.std():.3f}  (min: {losses.min().item():.3f}  max: {losses.max().item():.3f})')
        assert label not in epoch_train_losses and label not in test_accs
        test_accs[label] = res['test_accs']
        epoch_train_losses[label] = res['epoch_train_losses']

    # Training loss
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    ax.set_title(f'Training loss vs. Step (Rate: {rate} bits)')
    ax.set_ylim(0.3, 0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Training Loss')
    ax.set_xticks([100, 300, 500, 700])
    ax.set_xticklabels([100, 300, 500, 700])
    for label in labels:
        trl = train_losses[label][:max_step]
        plot_with_ci(torch.arange(len(trl)), trl, ax, label=label, smooth=smooth)
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'mnist_grad_train_loss.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Gradient L2 distortion
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    ax.set_title(f'Distortion vs. Step (Rate: {rate} bits)')
    ax.set_xlabel('Step')
    ax.set_ylabel('$\ell_2$ Distortion')
    for label in labels:
        y = mses[label][:max_step]
        plot_with_ci(torch.arange(len(y)), y, ax, label=label, smooth=smooth)
    ax.legend()
    fig.savefig(os.path.join(out_dir, 'mnist_grad_mse.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Train set loss per epoch
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Set Loss')
    for label, marker in zip(['Ours'] + baseline_labels, _MARKERS):
        if label == 'Ours':
            y = epoch_train_losses['Distributed'][1:max_epoch+1]
        else:
            y = epoch_train_losses[label][1:max_epoch+1]
        plot_with_ci(torch.arange(len(y))+1, y, ax, label=label, smooth=False, marker=marker)
    ax.legend()
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels([0, 5, 10, 15])
    fig.savefig(os.path.join(out_dir, 'mnist_grad_epoch_train_loss.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Train set loss per epoch (zoomed in)
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Set Loss')
    for label, marker in zip(['Ours'] + baseline_labels, _MARKERS):
        if label == 'Ours':
            y = epoch_train_losses['Distributed'][11:max_epoch+1]
        else:
            y = epoch_train_losses[label][11:max_epoch+1]
        plot_with_ci(torch.arange(len(y))+11, y, ax, label=label, smooth=False, marker=marker)
    ax.legend()
    ax.set_xticks([11, 12, 13, 14, 15, 16])
    ax.set_xticklabels([11, 12, 13, 14, 15, 16])
    ax.set_xlim(10.5, 16.5)
    fig.savefig(os.path.join(out_dir, 'mnist_grad_epoch_train_loss_zoomed.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Test set accuracy per epoch (vs. baseline)
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Set Accuracy')
    for label, marker in zip(['Ours'] + baseline_labels, _MARKERS):
        if label == 'Ours':
            y = test_accs['Distributed'][1:max_epoch+1]
        else:
            y = test_accs[label][1:max_epoch+1]
        plot_with_ci(torch.arange(len(y))+1, y, ax, label=label, smooth=False, marker=marker)
    ax.legend()
    ax.set_ylim(0.3, 0.90)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels([0, 5, 10, 15])
    fig.savefig(os.path.join(out_dir, 'mnist_grad_epoch_test_acc.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Test set accuracy per epoch (among VQVAEs)
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Set Accuracy')
    for label, marker in zip(labels, _MARKERS):
        y = test_accs[label][1:max_epoch+1]
        plot_with_ci(torch.arange(len(y))+1, y, ax, label=label, smooth=smooth)
    ax.legend()
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels([0, 5, 10, 15])
    fig.savefig(os.path.join(out_dir, 'mnist_grad_test_acc.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Test set loss per epoch
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    ax.set_title(f'Test Loss vs. Epoch (Rate: {rate} bits)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Set Loss')
    for label in labels:
        y = test_losses[label][1:max_epoch+1]
        plot_with_ci(torch.arange(len(y))+1, y, ax, label=label, smooth=False)
    ax.legend()
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels([0, 5, 10, 15])
    fig.savefig(os.path.join(out_dir, 'mnist_grad_test_loss.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_baseline_results(
    root_dir: str = 'grad_baseline_data'):
    fns = ['active_norm_k_clean', 'Q_clean', 'rand_k_clean', 'top_clean']
    names = ['CoordSample', 'QSGD', 'Random-$k$', 'Top-$k$']
    result = dict()
    for name, fn in zip(names, fns):
        with open(os.path.join(root_dir, fn), 'r') as f:
            dd = json.load(f)
        res = {
            'epoch_train_losses': torch.Tensor([x['train_loss'] for x in dd]).transpose(0, 1),
            'test_accs': torch.Tensor([x['test_acc'] for x in dd]).transpose(0, 1) / 100.,
        }
        result[name] = res
    return result


if __name__ == '__main__':
    fire.Fire({
        'prep': prepare_dataset,
        'gather_gradients': gather_gradients,
        'train_vqvae': train_vqvae,
        'eval': evaluate,
        'plot': plot,
        'plot_vqvae': plot_vqvae,
        'test_cnn': test_cnn,
    })
