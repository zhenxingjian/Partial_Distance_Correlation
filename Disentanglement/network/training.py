import os
import itertools
import pickle
import json
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from network.modules import BetaVAEGenerator, BetaVAEEncoder
from network.modules import StyleGenerator, ConvEncoder, ResidualEncoder, VGGDistance
from network.utils import ImageTensorDataset, Distance_Correlation

from evaluation import dci, sap, mig, classifier
from model import Discriminator  # from stylegan2


class MultiFactorClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.base_encoder = ConvEncoder(img_shape=config['img_shape'], max_conv_dim=128)
        self.heads = nn.ModuleList([
            nn.Linear(in_features=128, out_features=config['factor_sizes'][f])
            for f in range(config['n_factors'])
        ])

    def forward(self, img):
        x = self.base_encoder(img).reshape(img.shape[0], -1)
        return [head(x) for head in self.heads]


class FactorModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.factor_embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=config['factor_sizes'][f],
                embedding_dim=config['factor_dim'],
                _weight=(2 * torch.rand(config['factor_sizes'][f], config['factor_dim']) - 1) * 0.05
            )

            for f in range(config['n_factors'])
        ])

        if config['arch_betavae']:
            self.factor_classifiers = nn.ModuleList([
                BetaVAEEncoder(n_channels=config['img_shape'][-1], latent_dim=config['factor_sizes'][f])
                for f in range(config['n_factors'])
            ])

        else:
            self.factor_classifiers = MultiFactorClassifier(config)

    def forward(self, img, factors=None, label_masks=None):
        out = dict()

        factor_codes = []
        assignment_entropy = []

        if isinstance(self.factor_classifiers, MultiFactorClassifier):
            logits = self.factor_classifiers(img)
        else:
            logits = [factor_classifier(img) for factor_classifier in self.factor_classifiers]

        for f in range(self.config['n_factors']):
            assignment = Categorical(logits=logits[f])

            with torch.no_grad():
                factor_values = torch.arange(self.config['factor_sizes'][f], dtype=torch.int64).to(img.device)
                factor_embeddings = self.factor_embeddings[f](factor_values)

            if factors is not None:
                factor_code = (
                    self.factor_embeddings[f](factors[:, f]) * label_masks[:, [f]]
                    + torch.matmul(assignment.probs, factor_embeddings) * (~label_masks[:, [f]])
                )

            else:
                factor_code = torch.matmul(assignment.probs, factor_embeddings)

            factor_codes.append(factor_code)
            assignment_entropy.append(assignment.entropy())
            out['assignment_logits_{}'.format(f)] = assignment.logits

        out['factor_codes'] = torch.cat(factor_codes, dim=1)
        out['assignment_entropy'] = torch.stack(assignment_entropy, dim=1)

        return out


class LatentModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.factor_model = FactorModel(config)

        self.residual_embeddings = nn.Embedding(
            num_embeddings=config['n_imgs'],
            embedding_dim=config['residual_dim'],
            _weight=(2 * torch.rand(config['n_imgs'], config['residual_dim']) - 1) * 0.05
        )

        if config['arch_betavae']:
            self.generator = BetaVAEGenerator(
                latent_dim=config['n_factors'] * config['factor_dim'] + config['residual_dim'],
                n_channels=config['img_shape'][-1]
            )

        else:
            self.generator = StyleGenerator(
                latent_dim=config['n_factors'] * config['factor_dim'] + config['residual_dim'],
                img_size=config['img_shape'][0]
            )

        self.factor_model = torch.nn.DataParallel(self.factor_model)
        self.residual_embeddings = torch.nn.DataParallel(self.residual_embeddings)
        self.generator = torch.nn.DataParallel(self.generator)


class AmortizedModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.factor_model = FactorModel(config)

        if config['arch_betavae']:
            self.residual_encoder = BetaVAEEncoder(n_channels=config['img_shape'][-1], latent_dim=config['residual_dim'])
            self.generator = BetaVAEGenerator(
                latent_dim=config['n_factors'] * config['factor_dim'] + config['residual_dim'],
                n_channels=config['img_shape'][-1]
            )

        else:
            self.residual_encoder = ResidualEncoder(img_size=config['img_shape'][0], latent_dim=config['residual_dim'])
            self.generator = StyleGenerator(
                latent_dim=config['n_factors'] * config['factor_dim'] + config['residual_dim'],
                img_size=config['img_shape'][0]
            )

        self.factor_model = torch.nn.DataParallel(self.factor_model)
        self.residual_encoder = torch.nn.DataParallel(self.residual_encoder)
        self.generator = torch.nn.DataParallel(self.generator)

        if self.config['synthesis']['adversarial']:
            self.discriminator = Discriminator(size=config['img_shape'][0])
            self.discriminator = torch.nn.DataParallel(self.discriminator)


class Model:

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed=config['seed'])

        self.latent_model = None
        self.amortized_model = None

        if config['loss_reconstruction'] == 'bce':
            self.reconstruction_loss = nn.BCELoss()
        elif config['loss_reconstruction'] == 'l1':
            self.reconstruction_loss = nn.L1Loss()
        elif config['loss_reconstruction'] == 'mse':
            self.reconstruction_loss = nn.MSELoss()
        elif config['loss_reconstruction'] == 'perceptual':
            self.reconstruction_loss = VGGDistance(layer_ids=config['perceptual_loss'])
        else:
            raise Exception('unsupported reconstruction loss')

        self.classification_loss = nn.CrossEntropyLoss()
        self.visualization_rs = np.random.RandomState(seed=config['seed'])

    @staticmethod
    def load(model_dir):
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
            config = pickle.load(config_fd)

        model = Model(config)

        if os.path.exists(os.path.join(model_dir, 'latent.pth')):
            model.latent_model = LatentModel(config)
            model.latent_model.load_state_dict(torch.load(os.path.join(model_dir, 'latent.pth')))

        if os.path.exists(os.path.join(model_dir, 'amortized.pth')):
            model.amortized_model = AmortizedModel(config)
            model.amortized_model.load_state_dict(torch.load(os.path.join(model_dir, 'amortized.pth')))

        return model

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
            pickle.dump(self.config, config_fd)

        if self.latent_model:
            torch.save(self.latent_model.state_dict(), os.path.join(model_dir, 'latent.pth'))

        if self.amortized_model:
            torch.save(self.amortized_model.state_dict(), os.path.join(model_dir, 'amortized.pth'))

    def train_latent_model(self, imgs, factors, label_masks, residual_factors, model_dir, tensorboard_dir):
        self.latent_model = LatentModel(self.config)
        self.latent_model.load_state_dict(torch.load(os.path.join(model_dir, 'latent.pth')))

        data = dict(
            img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
            img_id=torch.from_numpy(np.arange(imgs.shape[0])),
            factors=torch.from_numpy(factors.astype(np.int64)),
            residual_factors=torch.from_numpy(residual_factors.astype(np.int64)),
            label_masks=torch.from_numpy(label_masks.astype(np.bool))
        )

        dataset = ImageTensorDataset(data)
        data_loader = DataLoader(
            dataset, batch_size=self.config['train']['batch_size'],
            shuffle=True, pin_memory=True, drop_last=False
        )

        label_ids = np.sum(label_masks, axis=1) > 0
        dataset_labeled = ImageTensorDataset({name: tensor[label_ids] for name, tensor in data.items()})
        data_loader_labeled = DataLoader(
            dataset_labeled, batch_size=self.config['train']['batch_size'],
            shuffle=True, pin_memory=True, drop_last=False
        )

        optimizer = Adam([
            {
                'params': itertools.chain(
                    self.latent_model.factor_model.module.factor_embeddings.parameters(),
                    self.latent_model.residual_embeddings.parameters()
                ),

                'lr': self.config['train']['learning_rate']['latent']
            },
            {
                'params': self.latent_model.factor_model.module.factor_classifiers.parameters(),
                'lr': self.config['train']['learning_rate']['classifier']
            },
            {
                'params': self.latent_model.generator.parameters(),
                'lr': self.config['train']['learning_rate']['generator']
            }
        ], betas=(0.5, 0.999))

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['train']['n_epochs'] * len(data_loader),
            eta_min=self.config['train']['learning_rate']['min']
        )

        self.latent_model.to(self.device)
        self.reconstruction_loss.to(self.device)

        summary = SummaryWriter(log_dir=tensorboard_dir)
        for epoch in range(self.config['train']['n_epochs'] ):
            self.latent_model.train()

            pbar = tqdm(iterable=data_loader)
            iterator_labeled = iter(data_loader_labeled)

            for batch in pbar:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

                losses_unsupervised = self.__iterate_latent_model(batch)
                loss_unsupervised = 0
                for term, val in losses_unsupervised.items():
                    if term == 'entropy' and epoch < self.config['train']['n_epochs_before_entropy']:
                        continue

                    loss_unsupervised += self.config['train']['loss_weights'][term] * val

                try:
                    batch_labeled = next(iterator_labeled)
                except StopIteration:
                    iterator_labeled = iter(data_loader_labeled)
                    batch_labeled = next(iterator_labeled)

                batch_labeled = {name: tensor.to(self.device) for name, tensor in batch_labeled.items()}

                losses_supervised = self.__iterate_latent_model_with_labels(batch_labeled)
                loss_supervised = 0
                for term, val in losses_supervised.items():
                    loss_supervised += self.config['train']['loss_weights'][term] * val

                loss = loss_unsupervised + self.config['train']['loss_weights']['supervised'] * loss_supervised

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_description_str('[disentanglement] epoch #{}'.format(epoch))
                pbar.set_postfix(loss=loss.item())

            pbar.close()

            summary.add_scalar(tag='loss/total', scalar_value=loss.item(), global_step=epoch)
            summary.add_scalar(tag='loss/unsupervised', scalar_value=loss_unsupervised.item(), global_step=epoch)
            summary.add_scalar(tag='loss/supervised', scalar_value=loss_supervised.item(), global_step=epoch)

            for term, val in losses_unsupervised.items():
                summary.add_scalar(tag='loss/unsupervised/{}'.format(term), scalar_value=val.item(), global_step=epoch)

            for term, val in losses_supervised.items():
                summary.add_scalar(tag='loss/supervised/{}'.format(term), scalar_value=val.item(), global_step=epoch)

            if epoch % self.config['train']['n_epochs_between_evals'] == 0 and self.config['gt_labels']:
                latent_factors = self.__embed_factors(dataset)
                scores = dci.evaluate(latent_factors, factors)

                summary.add_scalar(tag='dci/informativeness', scalar_value=scores['informativeness_test'], global_step=epoch)
                summary.add_scalar(tag='dci/disentanglement', scalar_value=scores['disentanglement'], global_step=epoch)
                summary.add_scalar(tag='dci/completeness', scalar_value=scores['completeness'], global_step=epoch)

                for factor_idx, factor_name in enumerate(self.config['factor_names']):
                    acc = self.__eval_factor_classification(imgs, factors, factor_idx)
                    summary.add_scalar(tag='factors/{}'.format(factor_name), scalar_value=acc, global_step=epoch)

                latent_residuals = self.__embed_residuals(dataset)
                for factor_idx, factor_name in enumerate(self.config['factor_names']):
                    acc_train, acc_test = classifier.logistic_regression(latent_residuals, factors[:, factor_idx])
                    summary.add_scalar(tag='residual/to-{}'.format(factor_name), scalar_value=acc_test, global_step=epoch)

                for factor_idx, factor_name in enumerate(self.config['residual_factor_names']):
                    acc_train, acc_test = classifier.logistic_regression(latent_residuals, residual_factors[:, factor_idx])
                    summary.add_scalar(tag='residual/to-{}'.format(factor_name), scalar_value=acc_test, global_step=epoch)

            if epoch % self.config['train']['n_epochs_between_visualizations'] == 0:
                figure = self.__visualize_reconstruction(dataset)
                summary.add_image(tag='reconstruction', img_tensor=figure, global_step=epoch)

                for factor_idx, factor_name in enumerate(self.config['factor_names']):
                    figure_fixed = self.__visualize_translation(dataset, factor_idx, randomized=False)
                    figure_random = self.__visualize_translation(dataset, factor_idx, randomized=True)

                    summary.add_image(tag='{}-fixed'.format(factor_name), img_tensor=figure_fixed, global_step=epoch)
                    summary.add_image(tag='{}-random'.format(factor_name), img_tensor=figure_random, global_step=epoch)

            self.save(model_dir)

        summary.close()

    def warmup_amortized_model(self, imgs, factors, label_masks, residual_factors, model_dir, tensorboard_dir):
        self.amortized_model = AmortizedModel(self.config)
        self.amortized_model.load_state_dict(torch.load(os.path.join(model_dir, 'amortized.pth')))
        self.amortized_model.factor_model.load_state_dict(self.latent_model.factor_model.state_dict())
        self.amortized_model.generator.load_state_dict(self.latent_model.generator.state_dict())

        data = dict(
            img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
            img_id=torch.from_numpy(np.arange(imgs.shape[0])),
            factors=torch.from_numpy(factors.astype(np.int64)),
            residual_factors=torch.from_numpy(residual_factors.astype(np.int64)),
            label_masks=torch.from_numpy(label_masks.astype(np.bool))
        )

        dataset = ImageTensorDataset(data)
        data_loader = DataLoader(
            dataset, batch_size=self.config['amortization']['batch_size'],
            shuffle=True, pin_memory=True, drop_last=False
        )

        optimizer = Adam(
            params=self.amortized_model.residual_encoder.parameters(),
            lr=self.config['amortization']['learning_rate']['max']
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['amortization']['n_epochs'] * len(data_loader),
            eta_min=self.config['amortization']['learning_rate']['min']
        )

        self.latent_model.to(self.device)
        self.amortized_model.to(self.device)

        os.mkdir(tensorboard_dir)
        summary = SummaryWriter(log_dir=tensorboard_dir)

        for epoch in range(self.config['amortization']['n_epochs']):
            self.latent_model.train()
            self.amortized_model.train()

            pbar = tqdm(iterable=data_loader)
            for batch in pbar:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

                losses = self.__iterate_encoders(batch)
                loss_total = 0
                for term, loss in losses.items():
                    loss_total += loss

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_description_str('[amortization] epoch #{}'.format(epoch))
                pbar.set_postfix(loss=loss_total.item())

            pbar.close()

            summary.add_scalar(tag='loss/encoders', scalar_value=loss_total.item(), global_step=epoch)

            for term, loss in losses.items():
                summary.add_scalar(tag='loss/encoders/{}'.format(term), scalar_value=loss.item(), global_step=epoch)

            if epoch % self.config['amortization']['n_epochs_between_evals'] == 0 and self.config['gt_labels']:
                latent_factors = self.__encode_factors(imgs)
                scores = dci.evaluate(latent_factors, factors)

                summary.add_scalar(tag='dci/informativeness', scalar_value=scores['informativeness_test'], global_step=epoch)
                summary.add_scalar(tag='dci/disentanglement', scalar_value=scores['disentanglement'], global_step=epoch)
                summary.add_scalar(tag='dci/completeness', scalar_value=scores['completeness'], global_step=epoch)

                latent_residuals = self.__encode_residuals(imgs)
                for factor_idx, factor_name in enumerate(self.config['factor_names']):
                    acc_train, acc_test = classifier.logistic_regression(latent_residuals, factors[:, factor_idx])
                    summary.add_scalar(tag='residual/to-{}'.format(factor_name), scalar_value=acc_test, global_step=epoch)

                for factor_idx, factor_name in enumerate(self.config['residual_factor_names']):
                    acc_train, acc_test = classifier.logistic_regression(latent_residuals, residual_factors[:, factor_idx])
                    summary.add_scalar(tag='residual/to-{}'.format(factor_name), scalar_value=acc_test, global_step=epoch)

            if epoch % self.config['amortization']['n_epochs_between_visualizations'] == 0:
                figure = self.__visualize_reconstruction(dataset, amortized=True)
                summary.add_image(tag='reconstruction', img_tensor=figure, global_step=epoch)

                for factor_idx, factor_name in enumerate(self.config['factor_names']):
                    figure_fixed = self.__visualize_translation(dataset, factor_idx, randomized=False, amortized=True)
                    figure_random = self.__visualize_translation(dataset, factor_idx, randomized=True, amortized=True)

                    summary.add_image(tag='{}-fixed'.format(factor_name), img_tensor=figure_fixed, global_step=epoch)
                    summary.add_image(tag='{}-random'.format(factor_name), img_tensor=figure_random, global_step=epoch)

            self.save(model_dir)

        summary.close()

    def tune_amortized_model(self, imgs, factors, label_masks, residual_factors, model_dir, tensorboard_dir):
        data = dict(
            img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
            img_id=torch.from_numpy(np.arange(imgs.shape[0])),
            factors=torch.from_numpy(factors.astype(np.int64)),
            residual_factors=torch.from_numpy(residual_factors.astype(np.int64)),
            label_masks=torch.from_numpy(label_masks.astype(np.bool))
        )

        dataset = ImageTensorDataset(data)
        data_loader = DataLoader(
            dataset, batch_size=self.config['synthesis']['batch_size'],
            shuffle=True, pin_memory=True, drop_last=True
        )

        generator_optimizer = Adam(
            params=itertools.chain(
                self.amortized_model.residual_encoder.parameters(),
                self.amortized_model.generator.parameters()
            ),

            lr=self.config['synthesis']['learning_rate']['generator'],
            betas=(0.5, 0.999)
        )

        if self.config['synthesis']['adversarial']:
            discriminator_optimizer = Adam(
                params=self.amortized_model.discriminator.parameters(),
                lr=self.config['synthesis']['learning_rate']['discriminator'],
                betas=(0.5, 0.999)
            )

        self.latent_model.to(self.device)
        self.amortized_model.to(self.device)
        self.reconstruction_loss.to(self.device)

        os.mkdir(tensorboard_dir)
        summary = SummaryWriter(log_dir=tensorboard_dir)

        for epoch in range(self.config['synthesis']['n_epochs'] + 1):
            self.latent_model.train()
            self.amortized_model.train()

            pbar = tqdm(iterable=data_loader)
            for batch in pbar:
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

                if self.config['synthesis']['adversarial']:
                    losses_discriminator = self.__iterate_discriminator(batch)
                    loss_discriminator = (
                        losses_discriminator['fake']
                        + losses_discriminator['real']
                        + losses_discriminator['gradient_penalty']
                    )

                    generator_optimizer.zero_grad()
                    discriminator_optimizer.zero_grad()
                    loss_discriminator.backward()
                    discriminator_optimizer.step()

                losses_generator = self.__iterate_amortized_model(batch)
                loss_generator = 0
                for term, loss in losses_generator.items():
                    loss_generator += self.config['synthesis']['loss_weights'][term] * loss

                generator_optimizer.zero_grad()
                if self.config['synthesis']['adversarial']:
                    discriminator_optimizer.zero_grad()

                loss_generator.backward()
                generator_optimizer.step()

                pbar.set_description_str('[synthesis] epoch #{}'.format(epoch))
                pbar.set_postfix(gen_loss=loss_generator.item())

            pbar.close()

            summary.add_scalar(tag='loss/generator', scalar_value=loss_generator.item(), global_step=epoch)
            if self.config['synthesis']['adversarial']:
                summary.add_scalar(tag='loss/discriminator', scalar_value=loss_discriminator.item(), global_step=epoch)

            for term, loss in losses_generator.items():
                summary.add_scalar(tag='loss/generator/{}'.format(term), scalar_value=loss.item(), global_step=epoch)

            if epoch % self.config['synthesis']['n_epochs_between_evals'] == 0 and self.config['gt_labels']:
                latent_residuals = self.__encode_residuals(imgs)
                for factor_idx, factor_name in enumerate(self.config['factor_names']):
                    acc_train, acc_test = classifier.logistic_regression(latent_residuals, factors[:, factor_idx])
                    summary.add_scalar(tag='residual/to-{}'.format(factor_name), scalar_value=acc_test, global_step=epoch)

                for factor_idx, factor_name in enumerate(self.config['residual_factor_names']):
                    acc_train, acc_test = classifier.logistic_regression(latent_residuals, residual_factors[:, factor_idx])
                    summary.add_scalar(tag='residual/to-{}'.format(factor_name), scalar_value=acc_test, global_step=epoch)

            if epoch % self.config['synthesis']['n_epochs_between_visualizations'] == 0:
                figure = self.__visualize_reconstruction(dataset, amortized=True)
                summary.add_image(tag='reconstruction', img_tensor=figure, global_step=epoch)

                for factor_idx, factor_name in enumerate(self.config['factor_names']):
                    figure_fixed = self.__visualize_translation(dataset, factor_idx, randomized=False, amortized=True)
                    figure_random = self.__visualize_translation(dataset, factor_idx, randomized=True, amortized=True)

                    summary.add_image(tag='{}-fixed'.format(factor_name), img_tensor=figure_fixed, global_step=epoch)
                    summary.add_image(tag='{}-random'.format(factor_name), img_tensor=figure_random, global_step=epoch)

            self.save(model_dir)

        summary.close()

    @torch.no_grad()
    def evaluate(self, imgs, factors, residual_factors, eval_dir):
        latent_factors = self.__encode_factors(imgs)

        scores = dci.evaluate(latent_factors, factors)
        with open(os.path.join(eval_dir, 'dci.json'), 'w') as fp:
            json.dump(scores, fp)

        scores = sap.evaluate(latent_factors, factors)
        with open(os.path.join(eval_dir, 'sap.json'), 'w') as fp:
            json.dump(scores, fp)

        scores = mig.evaluate(latent_factors, factors)
        with open(os.path.join(eval_dir, 'mig.json'), 'w') as fp:
            json.dump(scores, fp)

        scores = {}
        for f, factor_name in enumerate(self.config['factor_names']):
            scores[factor_name] = self.__eval_factor_classification(imgs, factors, f)

        with open(os.path.join(eval_dir, 'factors.json'), 'w') as fp:
            json.dump(scores, fp)

        latent_residuals = self.__encode_residuals(imgs)
        scores = {}
        for f, factor_name in enumerate(self.config['factor_names']):
            acc_train, acc_test = classifier.logistic_regression(latent_residuals, factors[:, f])
            scores[factor_name] = acc_test

        for f, factor_name in enumerate(self.config['residual_factor_names']):
            acc_train, acc_test = classifier.logistic_regression(latent_residuals, residual_factors[:, f])
            scores[factor_name] = acc_test

        with open(os.path.join(eval_dir, 'residual.json'), 'w') as fp:
            json.dump(scores, fp)

    # evaluate the distance correlation between residual factors and the attributes of interest
    @torch.no_grad()
    def evaluate_dc(self, imgs, factors, residual_factors, eval_dir, bert_embedding):
        latent_residuals = torch.from_numpy(self.__encode_residuals(imgs))
        factors_by_model = torch.from_numpy(self.__encode_factors(imgs))
        batch_size = 5000

        for factor_idx in range(6):
            pos = factors[:,factor_idx]>=0
            factor_select = factors[pos,factor_idx]
            latent_select = latent_residuals[pos,...]

            batch_num = len(factor_select) // batch_size
            # so that it won't happen x/0 issue
            if batch_num == 0:
                 batch_num = 1
            dc_factor = 0
            for batch_idx in range(batch_num):
                latent_batch = latent_select[batch_idx * batch_size: (batch_idx+1) * batch_size, ...]
                factor_batch = factor_select[batch_idx * batch_size: (batch_idx+1) * batch_size, ...]
                bert_embedding_batch = []
                for f in factor_batch:
                    bert_embedding_batch.append(bert_embedding[factor_idx][f])
                bert_embedding_batch = torch.cat(bert_embedding_batch)
                dc_factor += Distance_Correlation(latent_batch, bert_embedding_batch)
            dc_factor /= batch_num
            print(dc_factor)


        print('***** unsupervised *****')
        for factor_idx in range(6):
            # pos = factors[:,factor_idx]>=0
            factor_select = factors_by_model[:,factor_idx,:]
            latent_select = latent_residuals

            batch_num = len(latent_select) // batch_size
            dc_factor = 0
            for batch_idx in range(batch_num):
                latent_batch = latent_select[batch_idx * batch_size: (batch_idx+1) * batch_size, ...]
                factor_batch = factor_select[batch_idx * batch_size: (batch_idx+1) * batch_size, ...]

                dc_factor += Distance_Correlation(latent_batch, factor_batch)
            dc_factor /= batch_num
            print(dc_factor)
        


    @torch.no_grad()
    def manipulate(self, img, factor_name):
        self.amortized_model.to(self.device)
        self.amortized_model.eval()

        results = [img]

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        residual_code = self.amortized_model.residual_encoder(img.unsqueeze(dim=0))[0]

        factor_idx = self.config['factor_names'].index(factor_name)
        factor_codes = self.amortized_model.factor_model(img.unsqueeze(dim=0))['factor_codes'][0]

        factor_codes = list(torch.split(factor_codes, split_size_or_sections=self.config['factor_dim'], dim=0))
        factor_values = torch.arange(self.config['factor_sizes'][factor_idx], dtype=torch.int64).to(self.device)
        factor_embeddings = self.amortized_model.factor_model.module.factor_embeddings[factor_idx](factor_values)

        for v in range(factor_embeddings.shape[0]):
            factor_codes[factor_idx] = factor_embeddings[v]
            latent_code = torch.cat(factor_codes + [residual_code], dim=0)
            img_manipulated = self.amortized_model.generator(latent_code.unsqueeze(dim=0))[0]
            img_manipulated = (img_manipulated.clamp(min=0, max=1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            results.append(img_manipulated)

        return np.concatenate(results, axis=1)

    def __iterate_latent_model(self, batch):
        factor_model_out = self.latent_model.factor_model(batch['img'], batch['factors'], batch['label_masks'])
        residual_code = self.latent_model.residual_embeddings(batch['img_id'])

        if self.config['residual_std'] != 0:
            noise = torch.zeros_like(residual_code)
            noise.normal_(mean=0, std=self.config['residual_std'])

            residual_code_regularized = residual_code + noise
        else:
            residual_code_regularized = residual_code

        latent_code_regularized = torch.cat((factor_model_out['factor_codes'], residual_code_regularized), dim=1)
        img_reconstructed = self.latent_model.generator(latent_code_regularized)
        loss_reconstruction = self.reconstruction_loss(img_reconstructed, batch['img'])

        loss_entropy = factor_model_out['assignment_entropy'].mean()
        # Distance correlation to make the redisual code to be independent to the factors (attributes of interest)
        loss_residual_decay = Distance_Correlation(residual_code, factor_model_out['factor_codes'])

        return {
            'reconstruction': loss_reconstruction,
            'residual_decay': loss_residual_decay,
            'entropy': loss_entropy
        }

    def __iterate_latent_model_with_labels(self, batch):
        factor_model_out = self.latent_model.factor_model(batch['img'], batch['factors'], batch['label_masks'])
        residual_code = self.latent_model.residual_embeddings(batch['img_id'])

        if self.config['residual_std'] != 0:
            noise = torch.zeros_like(residual_code)
            noise.normal_(mean=0, std=self.config['residual_std'])

            residual_code_regularized = residual_code + noise
        else:
            residual_code_regularized = residual_code

        latent_code_regularized = torch.cat((factor_model_out['factor_codes'], residual_code_regularized), dim=1)
        img_reconstructed = self.latent_model.generator(latent_code_regularized)
        loss_reconstruction = self.reconstruction_loss(img_reconstructed, batch['img'])

        loss_classification = 0
        for f in range(self.config['n_factors']):
            if batch['label_masks'][:, f].any():
                loss_classification += self.classification_loss(
                    factor_model_out['assignment_logits_{}'.format(f)][batch['label_masks'][:, f]],
                    batch['factors'][batch['label_masks'][:, f], f]
                )

        # we follow the similar setup as the baseline paper
        # An Image is Worth More Than a Thousand Words: Towards Disentanglement in the Wild
        loss_residual_decay = Distance_Correlation(residual_code, factor_model_out['factor_codes'])

        return {
            'reconstruction': loss_reconstruction,
            'residual_decay': loss_residual_decay,
            'classification': loss_classification
        }

    def __iterate_encoders(self, batch):
        residual_code_target = self.latent_model.residual_embeddings(batch['img_id'])
        residual_code = self.amortized_model.residual_encoder(batch['img'])
        loss_residual = torch.mean((residual_code - residual_code_target) ** 2, dim=1).mean()

        return {
            'residual': loss_residual
        }

    def __iterate_amortized_model(self, batch):
        with torch.no_grad():
            factor_codes = self.amortized_model.factor_model(batch['img'])['factor_codes']
            residual_code_target = self.latent_model.residual_embeddings(batch['img_id'])

        residual_code = self.amortized_model.residual_encoder(batch['img'])

        latent_code = torch.cat((factor_codes, residual_code), dim=1)
        img_reconstructed = self.amortized_model.generator(latent_code)
        loss_reconstruction = self.reconstruction_loss(img_reconstructed, batch['img'])

        loss_residual = torch.mean((residual_code - residual_code_target) ** 2, dim=1).mean()

        losses = {
            'reconstruction': loss_reconstruction,
            'latent': loss_residual
        }

        if self.config['synthesis']['adversarial']:
            discriminator_fake = self.amortized_model.discriminator(img_reconstructed)
            loss_adversarial = self.__adv_loss(discriminator_fake, 1)

            losses['adversarial'] = loss_adversarial

        return losses

    def __iterate_discriminator(self, batch):
        with torch.no_grad():
            factor_codes = self.amortized_model.factor_model(batch['img'])['factor_codes']
            residual_code = self.amortized_model.residual_encoder(batch['img'])
            latent_code = torch.cat((factor_codes, residual_code), dim=1)
            img_reconstructed = self.amortized_model.generator(latent_code)

        batch['img'].requires_grad_()  # for gradient penalty
        discriminator_fake = self.amortized_model.discriminator(img_reconstructed)
        discriminator_real = self.amortized_model.discriminator(batch['img'])

        loss_fake = self.__adv_loss(discriminator_fake, 0)
        loss_real = self.__adv_loss(discriminator_real, 1)
        loss_gp = self.__gradient_penalty(discriminator_real, batch['img'])

        return {
            'fake': loss_fake,
            'real': loss_real,
            'gradient_penalty': loss_gp
        }

    @staticmethod
    def __adv_loss(logits, target):
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    @staticmethod
    def __gradient_penalty(d_out, x_in):
        batch_size = x_in.size(0)

        grad_dout = torch.autograd.grad(outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)

        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg

    @torch.no_grad()
    def __embed_factors(self, dataset):
        self.latent_model.eval()

        codes = []
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
        for batch in data_loader:
            batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

            batch_codes = self.latent_model.factor_model(batch['img'], batch['factors'], batch['label_masks'])['factor_codes']
            codes.append(batch_codes.cpu())

        codes = torch.cat(codes, dim=0)
        return torch.stack(torch.split(codes, split_size_or_sections=self.config['factor_dim'], dim=1), dim=1).numpy()

    @torch.no_grad()
    def __encode_factors(self, imgs):
        self.amortized_model.eval()

        codes = []
        dataset = ImageTensorDataset({'img': torch.from_numpy(imgs).permute(0, 3, 1, 2)})
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
        for batch in data_loader:
            batch_codes = self.amortized_model.factor_model(batch['img'].to(self.device))['factor_codes']
            codes.append(batch_codes.cpu())

        codes = torch.cat(codes, dim=0)
        return torch.stack(torch.split(codes, split_size_or_sections=self.config['factor_dim'], dim=1), dim=1).numpy()

    @torch.no_grad()
    def __embed_residuals(self, dataset):
        self.latent_model.eval()

        codes = []
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
        for batch in data_loader:
            batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

            batch_codes = self.latent_model.residual_embeddings(batch['img_id'])
            codes.append(batch_codes.cpu())

        codes = torch.cat(codes, dim=0)
        return codes.numpy()

    @torch.no_grad()
    def __encode_residuals(self, imgs):
        self.amortized_model.eval()

        codes = []
        dataset = ImageTensorDataset({'img': torch.from_numpy(imgs).permute(0, 3, 1, 2)})
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
        for batch in data_loader:
            batch_codes = self.amortized_model.residual_encoder(batch['img'].to(self.device))
            codes.append(batch_codes.cpu())

        codes = torch.cat(codes, dim=0)
        return codes.numpy()

    @torch.no_grad()
    def __eval_factor_classification(self, imgs, factors, factor_idx):
        self.latent_model.eval()

        dataset = ImageTensorDataset({'img': torch.from_numpy(imgs).permute(0, 3, 1, 2)})
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)

        predictions = []
        for batch in data_loader:
            if isinstance(self.latent_model.factor_model.module.factor_classifiers, MultiFactorClassifier):
                logits = self.latent_model.factor_model.module.factor_classifiers(batch['img'].to(self.device))[factor_idx]
            else:
                logits = self.latent_model.factor_model.module.factor_classifiers[factor_idx](batch['img'].to(self.device))

            batch_predictions = logits.argmax(dim=1)
            predictions.append(batch_predictions.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        accuracy = np.mean(factors[:, factor_idx] == predictions)

        return accuracy

    @torch.no_grad()
    def __visualize_translation(self, dataset, factor_idx, n_samples=10, randomized=False, amortized=False):
        random = self.visualization_rs if randomized else np.random.RandomState(seed=self.config['seed'])
        img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))
        batch = dataset[img_idx]
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

        if amortized:
            self.amortized_model.eval()

            batch['factor_codes'] = self.amortized_model.factor_model(batch['img'])['factor_codes']
            batch['residual_code'] = self.amortized_model.residual_encoder(batch['img'])

        else:
            self.latent_model.eval()

            batch['factor_codes'] = self.latent_model.factor_model(batch['img'], batch['factors'], batch['label_masks'])['factor_codes']
            batch['residual_code'] = self.latent_model.residual_embeddings(batch['img_id'])

        generator = self.amortized_model.generator if amortized else self.latent_model.generator

        figure = []
        for i in range(n_samples):
            converted_imgs = [batch['img'][i]]

            factor_codes = list(torch.split(batch['factor_codes'][i], split_size_or_sections=self.config['factor_dim'], dim=0))
            factor_values = torch.arange(self.config['factor_sizes'][factor_idx], dtype=torch.int64).to(self.device)
            factor_embeddings = self.latent_model.factor_model.module.factor_embeddings[factor_idx](factor_values)

            for j in range(factor_embeddings.shape[0]):
                factor_codes[factor_idx] = factor_embeddings[j]
                latent_code = torch.cat(factor_codes + [batch['residual_code'][i]], dim=0)
                converted_img = generator(latent_code.unsqueeze(dim=0))
                converted_imgs.append(converted_img[0])

            figure.append(torch.cat(converted_imgs, dim=2))

        figure = torch.cat(figure, dim=1)
        return figure.clamp(min=0, max=1)

    @torch.no_grad()
    def __visualize_reconstruction(self, dataset, n_samples=10, amortized=False):
        random = np.random.RandomState(seed=self.config['seed'])
        img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))
        batch = dataset[img_idx]
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

        if amortized:
            self.amortized_model.eval()

            batch['factor_codes'] = self.amortized_model.factor_model(batch['img'])['factor_codes']
            batch['residual_code'] = self.amortized_model.residual_encoder(batch['img'])

        else:
            self.latent_model.eval()

            batch['factor_codes'] = self.latent_model.factor_model(batch['img'], batch['factors'], batch['label_masks'])['factor_codes']
            batch['residual_code'] = self.latent_model.residual_embeddings(batch['img_id'])

        generator = self.amortized_model.generator if amortized else self.latent_model.generator

        latent_code = torch.cat((batch['factor_codes'], batch['residual_code']), dim=1)
        img_reconstructed = generator(latent_code)

        figure = torch.cat([
            torch.cat(list(batch['img']), dim=2),
            torch.cat(list(img_reconstructed), dim=2)
        ], dim=1)

        return figure.clamp(min=0, max=1)
