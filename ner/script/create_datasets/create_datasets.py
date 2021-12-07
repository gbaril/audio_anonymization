from pathlib import Path
from argparse import ArgumentParser
import os
from pprint import PrettyPrinter
from extract import extractDatasets
from parse import convertToDocbins, docbinToList, docbinToConll
import pickle


pprint = PrettyPrinter()

def pickleThis(dataset, name):
    with open(name, 'wb') as filehandle:
        pickle.dump(dataset, filehandle)

def writeThis(dataset, name):
    with open(name, 'w') as filehandle:
        filehandle.write(dataset)

# In ../data/spacy : Org, Loc, MoneyAmount, Currency and Person
# In ../data/spacy2 : All labels are Info
# With seed 120 :
# Number of entities {'total': 6308, 'train': 4152, 'dev': 903, 'test': 1253}
# Train size: 3097 , Dev size: 626 , Test size: 701

if __name__ == '__main__':
    this_file = Path(__file__)
    
    parser = ArgumentParser(this_file.name)
    parser.add_argument('--data', '-d', metavar='data_dir', default='/media/gbaril/Linux/Data/ner_french/data', help='Path to directory containing ann dataset')
    parser.add_argument('--output', '-o', metavar='output_dir', default='../../../data/ner', help='Path to output directory')
    parser.add_argument('--format', '-f', type=str, default='all', help='Format to output', choices=['spacy', 'hugging', 'conll', 'all'])
    parser.add_argument('--split', '-s', action='store_true', default=False)
    parser.add_argument('--verbose', '-v', type=int, default=0)
    args = parser.parse_args()

    output_dir = this_file.parent.joinpath(args.output).resolve()
    os.makedirs(output_dir, exist_ok=True)
    
    examples = extractDatasets(args.data, args.verbose)

    output = convertToDocbins(examples, args.verbose, args.split)

    if args.split:
        train, dev, test = output

        if args.format in ['spacy', 'all']:
            train.to_disk(output_dir.joinpath('train.spacy').resolve())
            dev.to_disk(output_dir.joinpath('dev.spacy').resolve())
            test.to_disk(output_dir.joinpath('test.spacy').resolve())
        
        if args.format in ['hugging', 'all']:
            train2 = docbinToList(train)
            pickleThis(train2, output_dir.joinpath('train.data').resolve())
            dev2 = docbinToList(dev)
            pickleThis(dev2, output_dir.joinpath('dev.data').resolve())
            test2 = docbinToList(test)
            pickleThis(test2, output_dir.joinpath('test.data').resolve())

        if args.format in ['conll', 'all']:
            train2 = docbinToConll(train)
            writeThis(train2, output_dir.joinpath('train.txt').resolve())
            dev2 = docbinToConll(dev)
            writeThis(dev2, output_dir.joinpath('dev.txt').resolve())
            test2 = docbinToConll(test)
            writeThis(test2, output_dir.joinpath('test.txt').resolve())

        print('Train size:', len(train), ', Dev size:', len(dev), ', Test size:', len(test))
    else:
        all = output[0]

        if args.format in ['spacy', 'all']:
            all.to_disk(output_dir.joinpath('all.spacy').resolve())
        
        if args.format in ['hugging', 'all']:
            all2 = docbinToList(all)
            pickleThis(all2, output_dir.joinpath('all.data').resolve())

        if args.format in ['conll', 'all']:
            all2 = docbinToConll(all)
            writeThis(all2, output_dir.joinpath('all.txt').resolve())

        print('Size: ', len(all))

    print('Saved in', output_dir)