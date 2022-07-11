import argparse
import os
import yaml
import imageio

import numpy as np
import cv2

import data
from assets import AssetManager
from network.training import Model

from transformers import BertTokenizer, BertModel

def BERT_embedding(values_names):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    Results = []

    for value_name in values_names:
        result = []
        for value in value_name:
            inputs = tokenizer(value, return_tensors="pt")
            outputs = model(**inputs)
            last_hidden_states = outputs[1]
            result.append(last_hidden_states)
        Results.append(result)

    return Results


def evaluate(args):
    assets = AssetManager(args.base_dir)
    model_dir = assets.get_model_dir(args.model_name)
    model = Model.load(model_dir)
    eval_dir = "MainPATH/eval/ffhq-x256/zerodim-ffhq-x256"
    os.mkdir(eval_dir)
    with open(os.path.join(os.path.dirname(__file__), 'config', '{}.yaml'.format(args.config)), 'r') as config_fp:
        config = yaml.safe_load(config_fp)


    data = np.load(assets.get_preprocess_file_path(args.data_name))
    imgs = data['imgs']

    labeled_factor_ids = [data['factor_names'].tolist().index(factor_name) for factor_name in config['factor_names']]
    residual_factor_ids = [f for f in range(len(data['factor_sizes'])) if f not in labeled_factor_ids]

    factors = data['factors'][:, labeled_factor_ids]
    residual_factors = data['factors'][:, residual_factor_ids]

    # breakpoint()
    # BERT embedding
    values_names = [["a kid","a teenager","an adult","an old person"],
                    ["a male","a female"],
                    ["a black person","a white person","an asian person"],
                    ["brunette hair","blond hair","bald","red hair","black hair", "white hair"],
                    ["a person with a beard","a person with a mustache","a person with a goatee","a shaved person"],
                    ["a person with glasses","a person with shades","a person without glasses"]]

    bert_embedding = BERT_embedding(values_names)


    model.evaluate_dc(imgs, factors, residual_factors, eval_dir, bert_embedding)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bd', '--base-dir', type=str, default='.')

    action_parsers = parser.add_subparsers(dest='action')
    action_parsers.required = True

    manipulate_parser = action_parsers.add_parser('manipulate')
    manipulate_parser.add_argument('-mn', '--model-name', type=str, required=True)
    manipulate_parser.add_argument('-fn', '--factor-name', type=str, required=True)
    manipulate_parser.add_argument('-i', '--img-path', type=str, required=True)
    manipulate_parser.add_argument('-o', '--output-img-path', type=str, required=True)
    manipulate_parser.set_defaults(func=manipulate)

    evaluate_parser = action_parsers.add_parser('evaluate')
    evaluate_parser.add_argument('-mn', '--model-name', type=str, required=True)
    evaluate_parser.add_argument('-dn', '--data-name', type=str, required=True)
    evaluate_parser.add_argument('-cf', '--config', type=str, required=True)
    evaluate_parser.set_defaults(func=evaluate)

    args, extras = parser.parse_known_args()
    if len(extras) == 0:
        args.func(args)
    else:
        args.func(args, extras)


if __name__ == '__main__':
    main()