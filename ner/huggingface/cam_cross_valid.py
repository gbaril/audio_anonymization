import sys
sys.path.append('/home/gbaril/Documents/ner/ner/script')

import os
import shutil
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, AdamW, get_scheduler, logging
from sklearn.model_selection import KFold
from evaluate import evaluate, info_label_transform
from sklearn.model_selection import train_test_split

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

def create_cross_file(fold, train_ids, test_ids, conll_examples):
    train_path = 'data/' + plus + 'ner/cross_train_' + str(fold) + '.txt'
    if os.path.exists(train_path):
        os.remove(train_path)
    with open(train_path, 'w') as filehandle:
        for id in train_ids:
            filehandle.write(conll_examples[id])

    test_path = 'data/' + plus + 'ner/cross_test_' + str(fold) + '.txt'
    if os.path.exists(test_path):
        os.remove(test_path)
    with open(test_path, 'w') as filehandle:
        for id in test_ids:
            filehandle.write(conll_examples[id])


# Evaluation function
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

# Create datasets from encodings and labels and create kfold object

dataset = Dataset(encodings, all_labels)
kfold = KFold(n_splits=folds, shuffle=True)

# Cross validation training
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print('-' * 30)
    print('Cross validation #', fold)
    
    CHECKPOINT_PATH = model_path + plus + model_path_name + '_' + str(fold) + '/'

    create_cross_file(fold, train_ids, test_ids, conll_examples)

    train_ids, dev_ids = train_test_split(train_ids, shuffle=True, random_state=0, test_size=0.1)

    # Create sampler and dataloader
    train_subsampler = SubsetRandomSampler(train_ids)
    dev_subsampler = SubsetRandomSampler(dev_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    dev_loader = DataLoader(dataset, batch_size=batch_size, sampler=dev_subsampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

    # Get and train the model
    model = CamembertForTokenClassification.from_pretrained(model_name, num_labels=len(labels))
    model.to(DEVICE)

    logs = []

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

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in train_loader:
            optim.zero_grad()

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            batch_labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
            
            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            optim.step()
            lr_scheduler.step()
         
        score, tscore = eval_on_dataset(model, dev_loader)
        epoch_loss = str(epoch_loss / len(dev_loader))
        
        print('[ Epoch', epoch, '] loss:', epoch_loss, 'train -> fscore:', score['total']['fscore'], 'precision:', score['total']['precision'], 'recall:', score['total']['recall'])
        logs.append({
            'epoch': epoch,
            'loss': epoch_loss,
            'dev_score': score,
            'dev_tscore': tscore
        })
        
        torch.save({
            'lr_scheduler': lr_scheduler.state_dict(),
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict()
        }, CHECKPOINT_PATH + plus + model_path_name + '_epoch_' + str(epoch) + '.pt')

    # Evaluate on test set
    checkpoint = torch.load(CHECKPOINT_PATH + plus + model_path_name + '_epoch_' + str(epochs - 1) + '.pt')
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    print('Without confidence and with random dropout on test set...')
    for i in range(5):
        score, tscore = eval_on_dataset(model, test_loader, confidence=None, verbose=True)
        logs.append({
            'test_score': score,
            'test_tscore': tscore,
            'mode': 'train',
            'num': i
        })

    model.eval()
    
    print('Eval Mode')
    print('Without confidence on test set...')
    score, tscore = eval_on_dataset(model, test_loader, confidence=None, verbose=True)
    logs.append({
        'test_score': score,
        'test_tscore': tscore,
        'mode': 'eval'
    })

    for conf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        print(f'With a confidence of {conf} on dev set...')
        score, tscore = eval_on_dataset(model, dev_loader, confidence=conf, verbose=True)
        logs.append({
            'confidence': conf,
            'dev_score': score,
            'dev_tscore': tscore,
            'mode': 'eval'
        })
        print(f'On test set...')
        score, tscore = eval_on_dataset(model, test_loader, confidence=conf, verbose=True)
        logs.append({
            'confidence': conf,
            'test_score': score,
            'test_tscore': tscore,
            'mode': 'eval'
        })

    with open(CHECKPOINT_PATH + 'logs', 'w') as fh:
            for log in logs:
                fh.write(f'{log}\n')

    torch.cuda.empty_cache()