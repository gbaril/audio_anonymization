import sys
sys.path.append('/home/gbaril/Documents/ner/ner/script')

import os
import shutil
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, AdamW, get_scheduler, logging
from tqdm.auto import tqdm
from evaluate import evaluate, info_label_transform

logging.set_verbosity_error()
torch.cuda.empty_cache()
#torch.manual_seed('0')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = '/home/gbaril/Documents/ner/ner/huggingface/'
model_name = 'camembert-base'
plus = '' # info_
model_path_name = 'model_cross_3'
train = False
epochs = 5
batch_size = 8
use_version = 2
confidence = 0.8
use_label_transform = False

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

# Read data from files
def read(filepath):
    with open(filepath, 'rb') as filehandle:
        dataset = pickle.load(filehandle)
    return dataset['tokens'], dataset['labels']

train_texts, train_labels = read('data/' + plus + 'ner/train.data')
val_texts, val_labels = read('data/' + plus + 'ner/dev.data')
test_texts, test_labels = read('data/' + plus + 'ner/all.data')

# Get labels
labels = sorted(list(set(tag for doc in train_labels for tag in doc)))
cleaned_labels = [l.split('-')[-1] for l in labels]
tag2id = {tag: id for id, tag in enumerate(labels)}
id2tag = {id: tag for tag, id in tag2id.items()}
O = cleaned_labels.index('O')

# Tokenize texts
tokenizer = CamembertTokenizerFast.from_pretrained(model_name)
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=False)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=False)
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=False)

# Encode labels
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

train_labels = encode_labels(train_labels, train_encodings)
val_labels = encode_labels(val_labels, val_encodings)
test_labels = encode_labels(test_labels, test_encodings)

# We don't want to pass this to the model
train_encodings.pop('offset_mapping')
val_encodings.pop('offset_mapping')
test_encodings.pop('offset_mapping')

# Create datasets from encodings and labels and then the loader
train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
test_dataset = Dataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Get and train the model
model = CamembertForTokenClassification.from_pretrained(model_name, num_labels=len(labels))
model.to(DEVICE)

def eval_on_dataset(model, loader, confidence=None, verbose=False):
    with torch.no_grad():
        preds = []
        golds = []

        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits.softmax(dim=-1).detach().cpu().clone().numpy()
            if confidence is not None:
                logits[logits[:,:,O] < confidence, O] = 0 # If confidence over O is less than X, remove it. We will choose an entity
            y_pred = logits.argmax(axis=-1)

            y_true = batch['labels'].detach().cpu().clone().numpy()

            for pred, gold_label in zip(y_pred, y_true):
                for p, g in zip(pred, gold_label):
                    if g != -100:
                        preds.append(id2tag[p])
                        golds.append(id2tag[g])

        if use_label_transform:
            score = evaluate(preds, golds, cleaned_labels, info_label_transform)
        else:
            score = evaluate(preds, golds, cleaned_labels)

        if verbose:
            print(score)

        return score['total']['fscore'], score['total']['precision'], score['total']['recall']

CHECKPOINT_PATH = model_path + plus + model_path_name + '/'

if train:
    if os.path.exists(CHECKPOINT_PATH):
        shutil.rmtree(CHECKPOINT_PATH)
    os.makedirs(CHECKPOINT_PATH)
    
    model.train()

    steps = epochs * len(train_loader)

    optim = AdamW(model.parameters(), lr=5e-5)

    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=steps
    )

    progress_bar = tqdm(range(steps))

    logs = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in train_loader:
            optim.zero_grad()

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            optim.step()
            lr_scheduler.step()
            progress_bar.update(1)
        
        fscore, precision, recall = eval_on_dataset(model, val_loader)
        epoch_loss = str(epoch_loss / len(train_loader))
        
        print('[ Epoch', epoch, '] loss:', epoch_loss, 'validation -> fscore:', fscore, 'precision:', precision, 'recall:', recall)
        logs.append({
            'epoch': epoch,
            'loss': epoch_loss,
            'val_fscore': fscore,
            'val_precision': precision,
            'val_recall': recall
        })
        
        torch.save({
            'lr_scheduler': lr_scheduler.state_dict(),
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict()
        }, CHECKPOINT_PATH + plus + model_path_name[:-2] + '_epoch_' + str(epoch) + '.pt')

    with open(CHECKPOINT_PATH + 'logs', 'w') as fh:
        for log in logs:
            fh.write(f'{log}\n')

# Evaluate on dev set
version = epochs - 1 if train else use_version
checkpoint = torch.load(CHECKPOINT_PATH + plus + model_path_name[:-2] + '_epoch_' + str(version) + '.pt')
    
model.load_state_dict(checkpoint['model_state_dict'])
model.train()

eval_on_dataset(model, test_loader, confidence=None, verbose=True)