import ast
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

file = open("imagenet1000_clsidx_to_labels.txt", "r")

contents = file.read()
dictionary = ast.literal_eval(contents)

class_embedding = []
for idx in range(1000):
    text = dictionary[idx]

    encoded_input = tokenizer(text.lower(), return_tensors='pt')
    output = model(**encoded_input)
    class_embedding.append(output[1])

class_embedding = torch.cat(class_embedding)

torch.save(class_embedding, 'ImageNet_Class_Embedding.pt')
