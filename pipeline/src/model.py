from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, logging
import spacy

logging.set_verbosity_error()
torch.cuda.empty_cache()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)
MODEL_NAME = 'camembert-base'
O = 10
BATCH_SIZE = 32
LABELS = ['B-Currency', 'B-Location', 'B-MoneyAmount', 'B-Organization', 'B-Person', 'I-Currency', 'I-Location', 'I-MoneyAmount', 'I-Organization', 'I-Person', 'O']
TAG2ID = {'B-Currency': 0, 'B-Location': 1, 'B-MoneyAmount': 2, 'B-Organization': 3, 'B-Person': 4, 'I-Currency': 5, 'I-Location': 6, 'I-MoneyAmount': 7, 'I-Organization': 8, 'I-Person': 9, 'O': 10}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, offsets):
        self.encodings = encodings
        self.offsets = offsets

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['offsets'] = torch.tensor(self.offsets[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def calculate_offsets(encodings):
    offsets = []
    for doc_offset, doc_encoding in zip(encodings.offset_mapping, encodings.encodings):
        # create an empty array of -100
        doc_enc = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        arr_token = np.array(doc_encoding.tokens)

        np.set_printoptions(threshold=np.inf)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0) & (arr_token != '▁')] = 1
        offsets.append(doc_enc.tolist())

    return offsets

def sentences_to_tokens(sentences: list):
    nlp = spacy.load("fr_dep_news_trf")
    tokens = []
    for sentence in sentences:
        doc = nlp.make_doc(sentence)
        tokens.append([str(s) for s in doc])
    return tokens

def create_dataloader(sentences: list, tokenize=True):
    if tokenize:
        tokens = sentences_to_tokens(sentences)
    else:
        tokens = sentences
    tokenizer = CamembertTokenizerFast.from_pretrained(MODEL_NAME)
    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=False)
    offsets = calculate_offsets(encodings)
    encodings.pop('offset_mapping')
    dataset = Dataset(encodings, offsets)
    return DataLoader(dataset, batch_size=BATCH_SIZE), tokens

class EnsembleModel:
    def __init__(self, model_dir: Path, n_labels: int = 11):
        self.model_dir = model_dir
        self.n_labels = n_labels
        self.models = []
        self._load_models()

    def _load_models(self):
        for model in self.model_dir.glob('*.pt'):
            checkpoint = torch.load(model)
            model = CamembertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=self.n_labels)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            self.models.append(model)

    def predict(self, loader: DataLoader, confidence = None):
        with torch.no_grad():
            preds = []

            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                offsets = batch['offsets'].to(DEVICE).detach().cpu().clone().numpy()

                logits = None
                
                for model in self.models:
                    outputs = model(input_ids, attention_mask=attention_mask)

                    if logits is None:
                        logits = outputs.logits
                    else:
                        logits += outputs.logits

                logits = logits.softmax(dim=-1).detach().cpu().clone().numpy()
                if confidence is not None:
                    logits[logits[:,:,O] < confidence, O] = 0 # If confidence over O is less than X, remove it. We will choose an entity
                #print(logits)
                y_pred = logits.argmax(axis=-1)

                for pred, offset in zip(y_pred, offsets):
                    preds.append([LABELS[p] for p, o in zip(pred, offset) if o != -100])
        
        return preds