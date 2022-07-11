# we follow the similar setup as the baseline paper:
# An Image is Worth More Than a Thousand Words: Towards Disentanglement in the Wild
# https://arxiv.org/abs/2106.15610

import argparse
import os
import yaml
import imageio

import numpy as np
import cv2

import data
from assets import AssetManager
from network.training import Model


def preprocess(args, extras=[]):
    assets = AssetManager(args.base_dir)

    img_dataset_def = data.supported_datasets[args.dataset_id]
    img_dataset = img_dataset_def(args.dataset_path, extras)

    np.savez(file=assets.get_preprocess_file_path(args.out_data_name), **img_dataset.read())


def train(args):
    assets = AssetManager(args.base_dir)
    model_dir = assets.recreate_model_dir(args.model_name)
    tensorboard_dir = assets.recreate_tensorboard_dir(args.data_name, args.model_name)
    eval_dir = assets.recreate_eval_dir(args.data_name, args.model_name)

    with open(os.path.join(os.path.dirname(__file__), 'config', '{}.yaml'.format(args.config)), 'r') as config_fp:
        config = yaml.safe_load(config_fp)

    data = np.load(assets.get_preprocess_file_path(args.data_name))
    imgs = data['imgs']

    labeled_factor_ids = [data['factor_names'].tolist().index(factor_name) for factor_name in config['factor_names']]
    residual_factor_ids = [f for f in range(len(data['factor_sizes'])) if f not in labeled_factor_ids]

    factors = data['factors'][:, labeled_factor_ids]
    residual_factors = data['factors'][:, residual_factor_ids]

    if config['gt_labels']:
        rs = np.random.RandomState(seed=args.seed)
        train_idx = rs.choice(imgs.shape[0], size=config['train_size'], replace=False)

        # labels are partial but complete = same seed for each factor
        rs = np.random.RandomState(seed=args.seed)
        label_idx = rs.choice(config['train_size'], size=config['n_labels_per_factor'], replace=False)

        label_masks = np.zeros_like(factors[train_idx]).astype(np.bool)
        for f in range(factors.shape[1]):
            label_masks[label_idx, f] = True
    else:
        train_idx = np.arange(imgs.shape[0])
        label_masks = np.zeros_like(factors[train_idx]).astype(np.bool)
        for f in range(factors.shape[1]):
            label_idx = (factors[:, f] != -1)
            label_masks[label_idx, f] = True
            factors[~label_idx, f] = 0  # dummy unused valid value

    config.update({
        'img_shape': imgs[train_idx].shape[1:],
        'n_imgs': imgs[train_idx].shape[0],
        'n_factors': len(labeled_factor_ids),
        'factor_sizes': data['factor_sizes'][labeled_factor_ids],
        'residual_factor_sizes': data['factor_sizes'][residual_factor_ids],
        'residual_factor_names': data['factor_names'][residual_factor_ids],
        'seed': args.seed
    })

    model = Model(config)
    model.train_latent_model(
        imgs[train_idx], factors[train_idx], label_masks, residual_factors[train_idx],
        model_dir, tensorboard_dir
    )

    model.warmup_amortized_model(
        imgs[train_idx], factors[train_idx], label_masks, residual_factors[train_idx],
        model_dir, tensorboard_dir=os.path.join(tensorboard_dir, 'amortization')
    )

    model.tune_amortized_model(
        imgs[train_idx], factors[train_idx], label_masks, residual_factors[train_idx],
        model_dir, tensorboard_dir=os.path.join(tensorboard_dir, 'synthesis')
    )

    if config['gt_labels']:
        model.evaluate(imgs, factors, residual_factors, eval_dir)


def manipulate(args):
    assets = AssetManager(args.base_dir)
    model_dir = assets.get_model_dir(args.model_name)
    model = Model.load(model_dir)

    img = imageio.imread(args.img_path)
    img = cv2.resize(img, dsize=(model.config['img_shape'][1], model.config['img_shape'][0]))
    manipulated_img = model.manipulate(img, args.factor_name)

    imageio.imwrite(args.output_img_path, manipulated_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bd', '--base-dir', type=str, default='.')

    action_parsers = parser.add_subparsers(dest='action')
    action_parsers.required = True

    preprocess_parser = action_parsers.add_parser('preprocess')
    preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=data.supported_datasets, required=True)
    preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
    preprocess_parser.add_argument('-odn', '--out-data-name', type=str, required=True)
    preprocess_parser.set_defaults(func=preprocess)

    train_parser = action_parsers.add_parser('train')
    train_parser.add_argument('-dn', '--data-name', type=str, required=True)
    train_parser.add_argument('-mn', '--model-name', type=str, required=True)
    train_parser.add_argument('-cf', '--config', type=str, required=True)
    train_parser.add_argument('-s', '--seed', type=int, default=0)
    train_parser.set_defaults(func=train)

    manipulate_parser = action_parsers.add_parser('manipulate')
    manipulate_parser.add_argument('-mn', '--model-name', type=str, required=True)
    manipulate_parser.add_argument('-fn', '--factor-name', type=str, required=True)
    manipulate_parser.add_argument('-i', '--img-path', type=str, required=True)
    manipulate_parser.add_argument('-o', '--output-img-path', type=str, required=True)
    manipulate_parser.set_defaults(func=manipulate)

    args, extras = parser.parse_known_args()
    if len(extras) == 0:
        args.func(args)
    else:
        args.func(args, extras)


if __name__ == '__main__':
    main()
