import argparse
from tqdm import tqdm

import numpy as np
import torch
import torchvision.transforms as transforms
import faiss

import clip
from clip.simple_tokenizer import SimpleTokenizer

from assets import AssetManager


def annotate_with_clip(args):
    assets = AssetManager(args.base_dir)
    data = np.load(assets.get_preprocess_file_path(args.data_name))
    imgs = data['imgs']

    att_names = ["age","gender","ethnicity","hair_color","beard","glasses"]
    values_names = [["a kid","a teenager","an adult","an old person"],
                    ["a male","a female"],
                    ["a black person","a white person","an asian person"],
                    ["brunette hair","blond hair","bald","red hair","black hair", "white hair"],
                    ["a person with a beard","a person with a mustache","a person with a goatee","a shaved person"],
                    ["a person with glasses","a person with shades","a person without glasses"]]

    model, preprocess = clip.load("ViT-B/32", device="cuda")
    model.eval()

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", model.context_length)
    print("Vocab size:", model.vocab_size)

    ENC_LEN = 512

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def get_emb(imgs, batch_size=32):
        set_length = len(imgs)
        img_enc = torch.zeros((set_length,ENC_LEN))

        run_ind = 0

        with torch.no_grad():
            for i in tqdm(range(int(np.ceil(set_length/batch_size)))):
                inputs = imgs[run_ind : run_ind + batch_size]
                inputs = [transform(img) for img in inputs]
                inputs = torch.stack(inputs)

                image_features = model.encode_image(inputs.cuda())
                img_enc[run_ind : run_ind + batch_size] = image_features.cpu()
                run_ind = run_ind + batch_size

        return img_enc

    img_enc = get_emb(imgs)
    img_enc /= img_enc.norm(dim=-1, keepdim=True)

    img_feat = img_enc.numpy()
    index = faiss.IndexFlatL2(img_feat.shape[1])
    index.add(np.ascontiguousarray(img_feat.astype('float32')))

    tokenizer = SimpleTokenizer()
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']

    values_list = np.full((len(att_names), img_feat.shape[0]), fill_value=-1)

    for i in range(len(att_names)):
        att = att_names[i]
        used_values = np.zeros(img_feat.shape[0])  # number of values assigned for an image (to prevent image having more than one value for an attribute)

        for j in range(len(values_names[i])):
            value = values_names[i][j]
            text_descriptions = f"%s"%(value)
            text_tokens = [[sot_token] + tokenizer.encode(text_descriptions) + [eot_token]]
            text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
            for k, tokens in enumerate(text_tokens):
                text_input[k, :len(tokens)] = torch.tensor(tokens)

            text_input = text_input.cuda()

            with torch.no_grad():
                text_features = model.encode_text(text_input).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)

            text_features = text_features.cpu().numpy()

            # find most compatible images
            D, I = index.search(np.ascontiguousarray(text_features.astype('float32')), args.K)
            I = list(I[0])
            I.sort()

            values_list[i, I] = j
            used_values[I] = used_values[I] + 1

        values_list[i, np.where(used_values>1)[0]] = -1  # ignore image attributes which were assigned with more than 1 value

    factors = values_list.T.astype(np.int64)

    np.savez(
        file=assets.get_preprocess_file_path(args.data_name),
        imgs=imgs,
        factors=factors,
        factor_sizes=[np.unique(factors[:, f]).size - 1 for f in range(factors.shape[1])],
        factor_names=att_names
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bd', '--base-dir', type=str, default='.')

    parser.add_argument('-dn', '--data-name', type=str, required=True)
    parser.add_argument('--K', type=int, default=1000)

    args = parser.parse_args()
    annotate_with_clip(args)

if __name__ == '__main__':
	main()
