import sys
sys.path.append('/home/gbaril/Documents/ner/ner/script')

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, logging
from evaluate import evaluate, info_label_transform

logging.set_verbosity_error()
torch.cuda.empty_cache() 
#torch.manual_seed(0)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = '/home/gbaril/Documents/ner/ner/huggingface/'
model_name = 'camembert-base'
plus = '' # info_
model_path_name = 'model_cross'
epochs = 5
folds = 10
batch_size = 16

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Evaluation function
def eval_on_dataset(models, loader, confidence=None, verbose=False):
        with torch.no_grad():
            preds = []
            golds = []

            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                logits = None
                
                for model in models:
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                    if logits is None:
                        logits = outputs.logits
                    else:
                        logits += outputs.logits

                logits = logits.softmax(dim=-1).detach().cpu().clone().numpy()
                if confidence is not None:
                    logits[logits[:,:,O] < confidence, O] = 0 # If confidence over O is less than X, remove it. We will choose an entity
                y_pred = logits.argmax(axis=-1)

                y_true = batch['labels'].detach().cpu().clone().numpy()

                for pred, gold_label in zip(y_pred, y_true):
                    for p, g in zip(pred, gold_label):
                        if g != -100:
                            preds.append(id2tag[p])
                            golds.append(id2tag[g])

            score = evaluate(preds, golds, cleaned_labels)
            tscore = evaluate(preds, golds, cleaned_labels, info_label_transform)

            if verbose:
                print('fscore:', score['total']['fscore'], 'precision:', score['total']['precision'], 'recall:', score['total']['recall'])

            return score, tscore

# Label encoding function
def encode_labels(labels, encodings):
    labels = [[tag2id[label] for label in doc] for doc in labels]
    encoded_labels = []
    for doc_labels, doc_offset, doc_encoding in zip(labels, encodings.offset_mapping, encodings.encodings):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        arr_token = np.array(doc_encoding.tokens)

        np.set_printoptions(threshold=np.inf)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0) & (arr_token != '▁')] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

# Read data function
def read(filepath):
    with open(filepath, 'rb') as filehandle:
        dataset = pickle.load(filehandle)
    return dataset['tokens'], dataset['labels']

# Read data from files
all_texts, all_labels = read('data/' + plus + 'ner/all.data')

# Get conll format files
conll_examples = []
with open('data/' + plus + 'ner/all.txt', 'r') as filehandle:
    example = ''
    for line in filehandle.readlines():
        example += line
        if line == '\n':
            conll_examples.append(example)
            example = ''

# Get labels
labels = sorted(list(set(tag for doc in all_labels for tag in doc)))
cleaned_labels = [l.split('-')[-1] for l in labels]
tag2id = {tag: id for id, tag in enumerate(labels)}
id2tag = {id: tag for tag, id in tag2id.items()}
O = tag2id['O']

# Tokenize texts
tokenizer = CamembertTokenizerFast.from_pretrained(model_name)
encodings = tokenizer(all_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=False)

# Encode labels
all_labels = encode_labels(all_labels, encodings)

# We don't want to pass this to the model
encodings.pop('offset_mapping')

# Create datasets from encodings and labels

dataset = Dataset(encodings, all_labels)
loader = DataLoader(dataset, batch_size=batch_size)

models = []

# Load models
for fold in range(10):
    print(f"Loading model {fold}...", end='')
    checkpoint_path = model_path + plus + model_path_name + '_' + str(fold) + '/'
    checkpoint = torch.load(checkpoint_path + plus + model_path_name + '_epoch_' + str(epochs - 1) + '.pt')

    model = CamembertForTokenClassification.from_pretrained(model_name, num_labels=len(labels))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    models.append(model)
    print("Done")

logs = []

score, tscore = eval_on_dataset(models, loader, confidence=0.9, verbose=True)

logs.append({
    'confidence': 0.9,
    'score': score,
    'tscore': tscore
})

score, tscore = eval_on_dataset(models, loader, verbose=True)

logs.append({
    'score': score,
    'tscore': tscore
})

with open(model_path + 'ensemble_logs', 'w') as fh:
    for log in logs:
        fh.write(f'{log}\n')

torch.cuda.empty_cache()