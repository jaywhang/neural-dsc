import argparse
import glob
import os
from typing import Dict

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
import seaborn as sns; sns.set(context='paper', style='white', font_scale=2.0, font='Times New Roman')

import numpy as np
import torch


_MARKERS = ['o', '^', 's', 'x', 'm']


def plot_rate_psnr(*, out_file: str, results: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    xticks = set()

    for i, (label, files) in enumerate(results.items()):
        rate_psnr = []
        for fn in files:
            res = torch.load(fn)
            if res['rate'] > 2048:
                continue
            rate_psnr.append((res['rate'], res['psnr'].mean().item()))
            xticks.add(res['rate'])
        rate_psnr = sorted(rate_psnr, key=lambda x: x[0])
        ax.plot(*zip(*rate_psnr), label=label, markersize=5, linewidth=1, marker=_MARKERS[i])

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bits)')
    ax.set_ylabel('PSNR')
    ax.set_ylim(41, 47.5)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def plot_rate_mse(*, out_file: str, results: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    xticks = set()

    for i, (label, files) in enumerate(results.items()):
        rate_mse = []
        for fn in files:
            res = torch.load(fn)
            if res['rate'] > 2048:
                continue
            rate_mse.append((res['rate'], res['mse'].mean().item()))
            xticks.add(res['rate'])
        rate_mse = sorted(rate_mse, key=lambda x: x[0])
        ax.plot(*zip(*rate_mse), label=label, markersize=5, linewidth=1, marker=_MARKERS[i])

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bits)')
    ax.set_ylabel('MSE')
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def plot_rate_ms_ssim(*, out_file: str, results: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    xticks = set()

    for i, (label, files) in enumerate(results.items()):
        rate_ms_ssim = []
        for fn in files:
            res = torch.load(fn)
            rate_ms_ssim.append((res['rate'], res['ms_ssim'].mean().item()))
            xticks.add(res['rate'])
        rate_ms_ssim = sorted(rate_ms_ssim, key=lambda x: x[0])
        ax.plot(*zip(*rate_ms_ssim), label=label, markersize=5, linewidth=1, marker=_MARKERS[i])

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bits)')
    ax.set_ylabel('MS-SSIM')
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)



def plot_bpp_ms_ssim(*, out_file: str, results: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    xticks = set()

    for i, (label, files) in enumerate(results.items()):
        bpp_ms_ssim = []
        for fn in files:
            res = torch.load(fn)
            bpp_ms_ssim.append((res['bpp'], res['ms_ssim'].mean().item()))
            xticks.add(res['bpp'])
        rate_mse = sorted(bpp_ms_ssim, key=lambda x: x[0])
        ax.plot(*zip(*bpp_ms_ssim), label=label, markersize=4, linewidth=1, marker=_MARKERS[i])

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('BPP')
    ax.set_ylabel('MS-SSIM')
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)



plot_rate_psnr(
    out_file='paper/celebahq256_top_rate_psnr.pdf',
    results={
        'Joint'         : glob.glob('checkpoints/celebahq256_vqvae_joint_[1234]bit/eval_*.pt'),
        'Distributed'   : glob.glob('checkpoints/celebahq256_vqvae_dist_[1234]bit/eval_*.pt'),
        'Separate'      : glob.glob('checkpoints/celebahq256_vqvae_separate_[1234]bit/eval_*.pt'),
    }
)

plot_rate_mse(
    out_file='paper/celebahq256_top_rate_mse.pdf',
    results={
        'Joint'         : glob.glob('checkpoints/celebahq256_vqvae_joint_[1234]bit/eval_*.pt'),
        'Distributed'   : glob.glob('checkpoints/celebahq256_vqvae_dist_[1234]bit/eval_*.pt'),
        'Separate'      : glob.glob('checkpoints/celebahq256_vqvae_separate_[1234]bit/eval_*.pt'),
    }
)

plot_rate_ms_ssim(
    out_file='paper/celebahq256_top_rate_ms_ssim.pdf',
    results={
        'Joint'         : glob.glob('checkpoints/celebahq256_vqvae_joint_[1234]bit/eval_*.pt'),
        'Distributed'   : glob.glob('checkpoints/celebahq256_vqvae_dist_[1234]bit/eval_*.pt'),
        'Separate'      : glob.glob('checkpoints/celebahq256_vqvae_separate_[1234]bit/eval_*.pt'),
    }
)

