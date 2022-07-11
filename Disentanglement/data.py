import argparse
import os
import glob
from abc import ABC, abstractmethod
from tqdm import tqdm

import numpy as np
import cv2
import scipy.io
import imageio
from sklearn.utils.extmath import cartesian
import h5py
import PIL


class DataSet(ABC):

    def __init__(self, base_dir=None, extras=None):
        super().__init__()
        self._base_dir = base_dir
        self._extras = extras

    @abstractmethod
    def read(self):
        pass

    def read_subset(self, size, seed):
        data = self.read()

        n_samples = data['imgs'].shape[0]
        rs = np.random.RandomState(seed)
        idx = rs.choice(n_samples, size, replace=False)

        data['imgs'] = data['imgs'][idx]
        data['factors'] = data['factors'][idx]

        return data


class Cars3D(DataSet):

    def __init__(self, base_dir, extras):
        super().__init__(base_dir, extras)

    def read(self):
        imgs = np.empty(shape=(4, 24, 183, 64, 64, 3), dtype=np.uint8)

        for i, filename in enumerate(glob.glob(os.path.join(self._base_dir, '*.mat'))):
            data_mesh = self._load_mesh(filename)

            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i, len(factor1) * len(factor2))
            ])

            imgs[all_factors[:, 0], all_factors[:, 1], all_factors[:, 2]] = data_mesh

        return {
            'imgs': np.reshape(imgs, (-1, 64, 64, 3)),
            'factors': cartesian((np.arange(4), np.arange(24), np.arange(183))),
            'factor_sizes': [4, 24, 183],
            'factor_names': ['elevation', 'azimuth', 'object']
        }

    @staticmethod
    def _load_mesh(filename):
        with open(os.path.join(filename), "rb") as f:
            mesh = np.einsum("abcde->deabc", scipy.io.loadmat(f)["im"])
        flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
        rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3), np.uint8)
        for i in range(flattened_mesh.shape[0]):
            pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
            pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
            rescaled_mesh[i, :, :, :] = np.array(pic)
        return rescaled_mesh


class SmallNorb(DataSet):

    def __init__(self, base_dir, extras):
        super().__init__(base_dir, extras)

        self.template = os.path.join(base_dir, "smallnorb-{}-{}.mat")
        self.chunk_names = [
            "5x46789x9x18x6x2x96x96-training",
            "5x01235x9x18x6x2x96x96-testing",
        ]

    def read(self):
        list_of_images, list_of_features = self._load_chunks(self.chunk_names)
        imgs = np.concatenate(list_of_images, axis=0)
        features = np.concatenate(list_of_features, axis=0)
        features[:, 3] = features[:, 3] // 2  # azimuth values are 0, 2, 4, ..., 24

        sort_idx = np.lexsort([features[:, i] for i in range(4, -1, -1)])

        return {
            'imgs': imgs[sort_idx],
            'factors': features[sort_idx],
            'factor_sizes': [np.unique(features[:, f]).size for f in range(features.shape[1])],
            'factor_names': ['category', 'instance', 'elevation', 'azimuth', 'lighting']
        }

    def _load_chunks(self, chunk_names):
        list_of_images = []
        list_of_features = []
        for chunk_name in chunk_names:
            norb = self._read_binary_matrix(self.template.format(chunk_name, "dat"))
            list_of_images.append(self._resize_images(norb[:, 0]))
            norb_class = self._read_binary_matrix(self.template.format(chunk_name, "cat"))
            norb_info = self._read_binary_matrix(self.template.format(chunk_name, "info"))
            list_of_features.append(np.column_stack((norb_class, norb_info)))

        return list_of_images, list_of_features

    @staticmethod
    def _read_binary_matrix(filename):
        with open(filename, "rb") as f:
            s = f.read()
            magic = int(np.frombuffer(s, "int32", 1))
            ndim = int(np.frombuffer(s, "int32", 1, 4))
            eff_dim = max(3, ndim)
            raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
            dims = []
            for i in range(0, ndim):
                dims.append(raw_dims[i])

            dtype_map = {
                507333717: "int8",
                507333716: "int32",
                507333713: "float",
                507333715: "double"
            }
            data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
        data = data.reshape(tuple(dims))
        return data

    @staticmethod
    def _resize_images(integer_images):
        resized_images = np.zeros((integer_images.shape[0], 64, 64, 1), dtype=np.uint8)
        for i in range(integer_images.shape[0]):
            image = PIL.Image.fromarray(integer_images[i, :, :])
            image = image.resize((64, 64), PIL.Image.ANTIALIAS)
            resized_images[i, :, :, 0] = image

        return resized_images


class Shapes3D(DataSet):

    def __init__(self, base_dir, extras):
        super().__init__(base_dir, extras)

        self.__data_path = os.path.join(base_dir, '3dshapes.h5')

    def read(self):
        with h5py.File(self.__data_path, 'r') as data:
            imgs = data['images'][:]
            labels = data['labels'][:]

            factors = np.zeros(labels.shape, dtype=np.int64)
            for f in range(labels.shape[1]):
                factor_unique_values = np.unique(labels[:, f])
                factors[:, f] = np.argmax(labels[:, [f]] == factor_unique_values, axis=1)

            return {
                'imgs': imgs,
                'factors': factors,
                'factor_sizes': [np.unique(factors[:, f]).size for f in range(factors.shape[1])],
                'factor_names': ['floor_color', 'wall_color', 'object_color', 'scale', 'shape', 'azimuth']
            }


class DSprites(DataSet):

    def __init__(self, base_dir, extras):
        super().__init__(base_dir, extras)

        self.__data_path = os.path.join(base_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    def read(self):
        data = np.load(self.__data_path)
        imgs = data['imgs'][..., np.newaxis] * 255
        factors = data['latents_classes'][:, 1:]

        return {
            'imgs': imgs,
            'factors': factors,
            'factor_sizes': [np.unique(factors[:, f]).size for f in range(factors.shape[1])],
            'factor_names': ['shape', 'scale', 'orientation', 'x', 'y']
        }


class FFHQ(DataSet):

    def __init__(self, base_dir, extras):
        super().__init__(base_dir)

        parser = argparse.ArgumentParser()
        parser.add_argument('-is', '--img-size', type=int, default=256)

        args = parser.parse_args(extras)
        self.__dict__.update(vars(args))

    def read(self):
        imgs = [] # np.empty(shape=(70000, self.img_size, self.img_size, 3), dtype=np.uint8)
        img_ids = np.arange(54001)
        for i in tqdm(img_ids):
            img_path = os.path.join(self._base_dir, 'imgs', '{:05d}.png'.format(i))
            if os.path.exists(img_path):
                imgs.append(cv2.resize(imageio.imread(img_path), dsize=(self.img_size, self.img_size)))
        imgs = np.stack(imgs)
        #breakpoint()
        return {
            'imgs': imgs
        }


supported_datasets = {
    'cars3d': Cars3D,
    'smallnorb': SmallNorb,
    'shapes3d': Shapes3D,
    'dsprites': DSprites,
    'ffhq': FFHQ
}
