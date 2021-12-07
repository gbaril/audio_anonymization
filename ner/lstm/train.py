#!/usr/bin/env python

import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
import json

from sklearn.model_selection import train_test_split
from utils import create_input
import loader

from utils import models_path, eval_script, eval_temp
import utils as u
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from model import Model

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-R", "--cross", default="0",
    help="cross-validation number"
)
optparser.add_option(
    "-Z", "--data", default="../../data/ner/",
    help="data path"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method

# Check parameters validity
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
models_path='./models/model_cross_' + opts.cross
print(models_path)
model = Model(parameters=parameters, models_path=models_path)
print ("Model location: %s" % model.model_path)

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(opts.data + 'cross_train_' + opts.cross + '.txt', lower, zeros)
test_sentences = loader.load_sentences(opts.data + 'cross_test_' + opts.cross + '.txt', lower, zeros)
train_sentences, dev_sentences = train_test_split(train_sentences, shuffle=True, random_state=0, test_size=0.1)

# Remove I, O, B before tags
def edit_tags(sentences):
    for s in sentences:
        for i in range(len(s)):
            newTag = s[i][-1].split('-')[-1]
            if newTag != 'O':
                newTag = 'B-' + newTag
            s[i][-1] = newTag

edit_tags(train_sentences)
edit_tags(test_sentences)
edit_tags(dev_sentences)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
O = tag_to_id['O']

# Index data
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

print ("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

# Save the mappings to disk
model.save_mappings(id_to_word, id_to_char, id_to_tag)

# Build the model
f_train, f_eval = model.build(**parameters)

# Reload previous model values
if opts.reload:
    print ('Reloading previous model...')
    model.reload()

#
# Train network
#
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])
n_epochs = 25  # number of epochs over the training set
best_dev = -np.inf

logs = []

for epoch in xrange(n_epochs):
    epoch_loss = 0

    print ("---------------------------------")
    print ("Starting epoch %i...") % epoch

    for i, index in enumerate(np.random.permutation(len(train_data))):
        input = create_input(train_data[index], parameters, True, singletons)
        new_cost = f_train(*input)
        epoch_loss += new_cost
    
    epoch_loss = epoch_loss / len(train_data)
    
    score, tscore = u.evaluate(parameters, f_eval, dev_sentences, dev_data, id_to_tag, dico_tags)

    logs.append({
        'epoch': epoch,
        'loss': epoch_loss,
        'dev_score': score,
        'dev_tscore': tscore
    })

    dev_f1score = score['total']['fscore']
    if dev_f1score > best_dev:
        best_dev = dev_f1score
        print ("New best score on dev.")
        print ("Saving model to disk...")
        model.save()

    print ("Epoch %i done." % epoch)
    print ("---------------------------------")

print('Evaluation on test dataset')
score, tscore = u.evaluate(parameters, f_eval, test_sentences, test_data, id_to_tag, dico_tags)
logs.append({
    'test_score': score,
    'test_tscore': tscore
})

with open('./models/model_cross_' + str(opts.cross) + '/log_' + str(opts.cross), 'w') as fh:
    for log in logs:
        fh.write(json.dumps(log)+'\n')

# for conf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
#     print 'With a confidence of ' + str(conf) + '...'
#     evaluate(parameters, f_eval, test_sentences, test_data, id_to_tag, dico_tags)