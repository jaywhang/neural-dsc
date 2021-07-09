import argparse
import functools
import glob
import os
import shutil
import sys
import tarfile
from zipfile import ZipFile

from PIL import Image
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm, trange


_DEFAULT_DATA_ROOT = 'data'


def _download_url(url, fpath):
    import urllib
    assert not os.path.exists(fpath)
    urllib.request.urlretrieve(url, fpath)


def _prepare_mnist(out_dir):
    temp = os.path.join(out_dir, 'mnist_temp')

    train = datasets.MNIST(root=temp, download=True, train=True)
    test = datasets.MNIST(root=temp, download=True, train=False)

    train_data = train.data.numpy()[..., np.newaxis]
    train_label = train.targets.numpy()
    test_data = test.data.numpy()[..., np.newaxis]
    test_label = test.targets.numpy()

    assert train_data.shape == (60000, 28, 28, 1) and test_data.shape == (10000, 28, 28, 1)
    assert train_data.dtype == test_data.dtype == np.uint8
    assert train_label.shape == (60000,) and test_label.shape == (10000,)
    assert train_label.dtype == test_label.dtype == np.int64

    np.save(os.path.join(out_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(out_dir, 'train_label.npy'), train_label)
    np.save(os.path.join(out_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(out_dir, 'test_label.npy'), test_label)

    shutil.rmtree(temp)
    print('MNIST dataset successfully created.')


def _prepare_celebahq(out_dir, *, size):
    if os.path.exists(os.path.join(out_dir, f'train_{size}x{size}.npy')):
        print(f'CelebA-HQ train_{size}x{size} already exists!')
        return

    import tensorflow as tf
    tar_path = os.path.join(out_dir, 'celeba-tfr.tar')
    tfr_path = os.path.join(out_dir, 'celeba-tfr')
    if not os.path.exists(tfr_path):
        if not os.path.exists(tar_path):
            print(f'Downloading CelebA-HQ tar archive...')
            TFR_URL = 'https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar'
            _download_url(TFR_URL, tar_path)
        assert os.path.exists(tar_path)

        tar_file = tarfile.open(tar_path)
        tar_file.extractall(out_dir)
    assert os.path.isdir(tfr_path) and len(os.listdir(tfr_path)) == 2

    def _resize_image_array(images, size, interpolation=Image.BILINEAR):
        assert type(size) == tuple
        N, origH, origW, C = images.shape  # Assume NHWC
        assert C in (1, 3) and images.dtype == np.uint8

        if size == (origH, origW):
            return images

        resized = []
        for img in images:
            pil = Image.fromarray(img.astype('uint8'), 'RGB')
            pil = pil.resize(size, resample=interpolation)
            resized.append(np.array(pil))

        resized = np.stack(resized, axis=0)
        assert resized.shape == (N, *size, C)

        return resized

    def _process_folder(split):
        split_str = {'train': 'train', 'val': 'validation'}[split]
        filenames = glob.glob(os.path.join(tfr_path, split_str, '*.tfrecords'))
        dataset = tf.data.TFRecordDataset(filenames=filenames)
        processed = []

        for example in tqdm(dataset, desc=f'Processing {split} set...'):
            parsed = tf.train.Example.FromString(example.numpy())
            shape = parsed.features.feature['shape'].int64_list.value
            img_bytes = parsed.features.feature['data'].bytes_list.value[0]
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)
            processed.append(img)

        processed = np.stack(processed)
        assert len(processed) == {'train': 27000, 'val': 3000}[split]
        assert processed.shape[1:] == (256, 256, 3)

        out_path = os.path.join(out_dir, f'{split}_{size}x{size}.npy')
        resized = _resize_image_array(processed, (size, size))
        assert resized.shape == (processed.shape[0], size, size, processed.shape[3])
        np.save(out_path, resized)
        print(f'Saved {out_path}')

    _process_folder('train')
    _process_folder('val')
    print(f'CelebA-HQ {size}x{size} successfully created.')


class MNIST(Dataset):
    def __init__(self, *, split, data_root):
        assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
        if split == 'val':
            print(f'INFO: MNIST does not have an official validation set. Using test set instead.')
            split = 'test'

        self.split = split
        self.image_shape = (1, 28, 28)
        self.data_root = data_root

        self.data = np.load(os.path.join(self.data_root, f'mnist/{self.split}_data.npy'))
        self.data = torch.from_numpy(self.data.transpose(0,3,1,2))
        self.label = np.load(os.path.join(self.data_root, f'mnist/{self.split}_label.npy'))
        self.label = torch.from_numpy(self.label)
        assert len(self.data) == len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class _CelebAHQBase(Dataset):
    image_size = None

    def __init__(self, *, split, data_root):
        assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
        if split == 'test':
            print(f'INFO: CelebaHQ does not have an official test set. Using val set instead.')
            split = 'val'

        self.split = split
        self.image_shape = (3, self.image_size, self.image_size)
        self.data_root = data_root
        self.data = np.load(os.path.join(data_root, f'celebahq/{split}_{self.image_size}x{self.image_size}.npy'))
        self.data = torch.from_numpy(self.data.transpose(0,3,1,2))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class CelebAHQ64(_CelebAHQBase):
    image_size = 64


class CelebAHQ128(_CelebAHQBase):
    image_size = 128


class CelebAHQ256(_CelebAHQBase):
    image_size = 256


##### Main methods #####

def prepare_dataset(dataset: str, data_root: str = _DEFAULT_DATA_ROOT):
    assert dataset is not None and data_root is not None
    mapping = {
        'mnist': _prepare_mnist,
        'celebahq64': functools.partial(_prepare_celebahq, size=64),
        'celebahq128': functools.partial(_prepare_celebahq, size=128),
        'celebahq256': functools.partial(_prepare_celebahq, size=256),
    }
    if dataset not in mapping:
        raise ValueError(f'Invalid dataset name {dataset}')
    if dataset.startswith('celebahq'):
        out_dir = os.path.join(data_root, 'celebahq')
    else:
        out_dir = os.path.join(data_root, dataset)

    os.makedirs(out_dir, exist_ok=True)
    mapping[dataset](out_dir)


def load_dataset(dataset: str, *args, data_root: bool = None, **kwargs):
    if data_root is None:
        data_root = _DEFAULT_DATA_ROOT
    kwargs['data_root'] = data_root

    return {
        'mnist': MNIST,
        'celebahq64': CelebAHQ64,
        'celebahq128': CelebAHQ128,
        'celebahq256': CelebAHQ256,
    }[dataset](*args, **kwargs)


