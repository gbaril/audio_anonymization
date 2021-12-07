from glob import iglob, glob
from pathlib import Path
from argparse import ArgumentParser
from json import loads
import decimal
import ast
import numpy as np
import matplotlib.pyplot as plt

def plot_meval(pop, mean, label='Precision', position='lower right'):
    #label_pop = 'Sam'
    #label_sam = 'Sub'

    p = plt.plot(pop, mean, alpha=0.7)
    #plt.errorbar(pop, mean, yerr=std, linestyle='None', fmt='.', label=label_sam, alpha=0.4, c=p[0].get_color())
    plt.scatter(pop, mean, c=p[0].get_color(), label=label, s=7, alpha=0.9)
    
    plt.ylabel('Score')
    plt.xlabel('Confidence threshold')
    plt.yticks(np.arange(0, 1.01, 0.1))
    #plt.xticks([x for x in pop if x % 2 == 0])
    
    plt.grid(True, axis='y', ls='--', alpha=0.4)
    plt.legend(loc=position)

def generate_table(title, first_column, first_row, values, skips=None, label=None, zero_row=None, first_row_bold=False):
    skip_i = 0
    if skips == None:
        skips = ['\\hline' for _ in range(len(values) + 20)]

    # First column emphasis if necessary
    if len(first_column) == len(values):
        for i, value in enumerate(zip(first_column, values)):
            value[1].insert(0, value[0])

    table = "\\begin{table}\n\\centering\n\\parbox{1.0\\textwidth}{\caption{" + title + "}\\label{tbl:"
    table += "LABELTODO" if label is None else label
    table += "}}\n\\begin{tabular}{"

    # Header
    first = ''
    if len(first_row) < len(values[0]):
        table += '|c'
        first += '& '


    for value in first_row:
        table += '|c'
        if first_row_bold:
            first += '{\\bf ' + str(value) + '} & '
        else:
            first += str(value) + ' & '
    first = first[:-2] + '\\\\\n' # remove last &
    table += '|}\n'
    
    if zero_row is not None:
        zero = ''
        for value in zero_row:
            zero += str(value) + ' & '
        zero = zero[:-2] + '\\\\\n' # remove last &
        table += '    \\hline\n        ' + zero

    table += '    \\hline\n        ' + first
        
    for value in values:
        first = ''
        for v in value:
            first += str(v) + ' & '
        first = first[:-2] + '\\\\\n' # remove last &
        table += '    ' + skips[skip_i] + '\n        ' + first
        skip_i += 1
    table += '    ' + skips[skip_i] + '\n\\end{tabular}\n\\end{table}'
    skip_i += 1
    
    return table

def round_float(s): return round(float(s), 3)

def bold(s): return ['{\\bf ' + i + '}' for i in s]

def dev_this(first_row, line_name, dev, special=''):
    all_data = [[[] for _ in range(4)] for __ in range(len(dev['0']))]

    for cross, lines in dev.items():
        for line in lines:
            epoch = line['epoch']

            all_data[epoch][0].append(float(line['loss']))
            all_data[epoch][1].append(line[line_name]['total']['precision'])
            all_data[epoch][2].append(line[line_name]['total']['recall'])
            all_data[epoch][3].append(line[line_name]['total']['fscore'])

    values = []
    first_column = []
    precision, recall, f1 = [], [], []
    for epoch, data in enumerate(all_data):
        first_column.append(epoch)
        values.append([f'{round(np.mean(data[i]), 3)} ± {round(np.std(data[i]), 3)}' for i in range(4)])
        
        precision.append(round(np.mean(data[1]), 3))
        recall.append(round(np.mean(data[2]), 3))
        f1.append(round(np.mean(data[3]), 3))
    
    plot_meval(first_column, precision, special + 'Precision')
    plot_meval(first_column, recall, special + 'Recall')
    plot_meval(first_column, f1, special + 'F1 score')
    print(generate_table(args.title, first_column, first_row, values), end='\n\n')

def test_this(first_row, line_name, test, special=''):

    convert = {'total': 'Total', 'INFO': 'NTE'}
    skip = ['O']

    all_data = {}

    for lines in test.values():
        for line in lines:
            for entity, data in line[line_name].items():
                if entity not in all_data:
                    all_data[entity] = [[] for _ in range(3)]
                all_data[entity][0].append(data['precision'])
                all_data[entity][1].append(data['recall'])
                all_data[entity][2].append(data['fscore'])

    values = []
    first_column = []

    for entity, data in all_data.items():
        if entity in convert:
            entity = convert[entity]
        if entity not in skip:
            first_column.append(entity)
            values.append([f'{round(np.mean(data[i]), 3)} ± {round(np.std(data[i]), 3)}' for i in range(3)])
    
    print(generate_table(args.title, first_column, first_row, values), end='\n\n')

def conf_this(first_row, line_name, test, special='', confidence=None):
    all_data = {}
    convert = {'total': 'Total', 'INFO': 'NTE'}
    skip = ['O']

    for lines in test.values():
        for line in lines:
            conf = line['confidence']
            if confidence == None:
                if conf not in all_data:
                    all_data[conf] = [[] for _ in range(3)]
                all_data[conf][0].append(line[line_name]['total']['precision'])
                all_data[conf][1].append(line[line_name]['total']['recall'])
                all_data[conf][2].append(line[line_name]['total']['fscore'])
            elif conf == confidence:
                for entity, data in line[line_name].items():
                    if entity not in all_data:
                        all_data[entity] = [[] for _ in range(3)]
                    all_data[entity][0].append(data['precision'])
                    all_data[entity][1].append(data['recall'])
                    all_data[entity][2].append(data['fscore'])

    values = []
    first_column = []

    precision, recall, f1 = [], [], []
    for conf, data in all_data.items():
        if conf in convert:
            conf = convert[conf]
        if conf not in skip:
            first_column.append(conf)
            values.append([f'{round(np.mean(data[i]), 3)} ± {round(np.std(data[i]), 3)}' for i in range(3)])

        precision.append(round(np.mean(data[0]), 3))
        recall.append(round(np.mean(data[1]), 3))
        f1.append(round(np.mean(data[2]), 3))
    
    if confidence == None:
        plot_meval(first_column, precision, special + 'Precision', position='lower left')
        plot_meval(first_column, recall, special + 'Recall', position='lower left')
        plot_meval(first_column, f1, special + 'F1 score', position='lower left')
        plt.show()
    print(generate_table(args.title, first_column, first_row, values), end='\n\n')


if __name__ == '__main__':
    this_file = Path(__file__)
    
    parser = ArgumentParser(this_file.name)
    parser.add_argument('--directory', '-d', metavar='data_dir', help='Path to directory containing logs')
    parser.add_argument('--format', '-f', type=str, default='log', help='Format of log files')
    parser.add_argument('--output', '-o', type=str, default='epochs', choices=['epochs','dev', 'test', 'conf_dev', 'conf_test'], help='Format of log files')
    parser.add_argument('--title', '-t', type=str, default='', help='Title of the table (if applicable)')
    parser.add_argument('--label', '-l', type=str, default=None, help='Label of the table (if applicable)')
    parser.add_argument('--prefix', '-p', type=str, default='dev_', help='prefix before score in logs')
    parser.add_argument('--confidence', '-c', type=str, default=0.9, help='confidence threshold')
    args = parser.parse_args()

    log_dir = this_file.parent.joinpath(args.directory).resolve()

    confidence_dev = {}
    dev = {}
    test = {}
    confidence_test = {}
    

    for path in sorted(glob(str(log_dir) + '/model_cross_*')):
        num = path[-1]
        confidence_dev[num] = []
        test[num] = []
        dev[num] = []
        confidence_test[num] = []
        for txt in iglob(path + '/' + args.format + '*'):
            with open(txt, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    data = loads(line.replace("'", '"'), parse_float=round_float)

                    if 'epoch' in data:
                        dev[num].append(data)
                    elif 'test_score' in data and not 'confidence' in data:
                        test[num].append(data)
                    elif 'test_score' in data:
                        confidence_test[num].append(data)
                    else:
                        confidence_dev[num].append(data)

    if args.output == 'epochs':
        zero_row = ['\multicolumn{2}{|c|}{}', '\multicolumn{3}{|c|}{\\textbf{Standard}}', '\multicolumn{3}{|c|}{\\textbf{No Type Error}}']
        first_row = bold(['Epoch', 'Loss', 'Precision', 'Recall', 'F1 Score', 'Precision', 'Recall', 'F1 Score'])

        for num, lines in dev.items():
            num_lines = str(len(lines))
            real_num = str(int(num) + 1)
            first_column = []
            values = []
            skips = ['\\hline' for _ in  range(len(lines) + 1)]

            for line in lines:
                first_column.append(line['epoch'])
                values.append([ 
                    round(float(line['loss']), 5), 
                    line[args.prefix + 'score']['total']['precision'], 
                    line[args.prefix + 'score']['total']['recall'], 
                    line[args.prefix + 'score']['total']['fscore'],
                    line[args.prefix + 'tscore']['total']['precision'], 
                    line[args.prefix + 'tscore']['total']['recall'], 
                    line[args.prefix + 'tscore']['total']['fscore']])
            print(generate_table(args.title + ' ' + real_num, 
            first_column, 
            first_row, 
            values, 
            skips,
            zero_row=zero_row, 
            label=args.label + '_' + real_num), end='\n\n')

    elif args.output == 'dev':
        dev_this(bold(['Epoch', 'Loss', 'Precision', 'Recall', 'F1 Score']), args.prefix + 'score', dev)
        dev_this(bold(['Epoch', 'Loss', 'NTE-Precision', 'NTE-Recall', 'NTE-F1 Score']), args.prefix + 'tscore', dev, special='NTE ')
        plt.show()

    elif args.output == 'test':
        test_this(bold(['Entity Type', 'Precision', 'Recall', 'F1 Score']), 'test_score', test)
        test_this(bold(['Entity Type', 'NTE-Precision', 'NTE-Recall', 'NTE-F1 Score']), 'test_tscore', test, special='NTE ')
        plt.show()
    elif args.output == 'conf_dev':
        conf_this(bold(['Confidence threshold', 'Precision', 'Recall', 'F1 Score']), 'dev_score', confidence_dev)
        conf_this(bold(['Confidence threshold', 'NTE-Precision', 'NTE-Recall', 'NTE-F1 Score']), 'dev_tscore', confidence_dev, special='NTE ')
        plt.show()
    elif args.output == 'conf_test':
        conf_this(bold(['Confidence threshold', 'Precision', 'Recall', 'F1 Score']), 'test_score', confidence_test, confidence=args.confidence)
        conf_this(bold(['Confidence threshold', 'NTE-Precision', 'NTE-Recall', 'NTE-F1 Score']), 'test_tscore', confidence_test, special='NTE ', confidence=args.confidence)
        plt.show()

# \begin{table}
# \centering
# \parbox{1.0\textwidth}{\caption{Transformer training result - Fold number 9}\label{tbl:allo_9}}
# \begin{tabular}{|c|c|c|c|c|c|}
#     \hline
#         {\bf Fold} & {\bf Epoch} & {\bf Loss} & {\bf Precision} & {\bf Recall} & {\bf F1 Score} \\
#     \hline
#         \multirow{5}{*}{9} & 0 & 0.70352 & 0.771 & 0.406 & 0.532 \\
#     \cline{2-6}
#           & 1 & 0.23008 & 0.91 & 0.872 & 0.89 \\
#     \cline{2-6}
#           & 2 & 0.1309 & 0.931 & 0.929 & 0.93 \\
#     \cline{2-6}
#           & 3 & 0.09834 & 0.936 & 0.953 & 0.945 \\
#     \cline{2-6}
#           & 4 & 0.08441 & 0.94 & 0.964 & 0.952 \\
#     \hline
# \end{tabular}
# \end{table}