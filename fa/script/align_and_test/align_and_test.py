from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from time import time

from mfa_utils import align_mfa
from sppas_utils import align_sppas
from evaluate_utils import single_eval, multiple_pop_eval
from plot import plot_meval, plot_mtolerance, generate_table

def eval(align, args):
    output_dir = data_dir.joinpath('output_' + args.true_algo + '/').resolve()
    gold_standard_dir = data_dir.joinpath('gold/').resolve()

    if not (output_dir.exists() and args.eval):
        if args.timer:
            start = time()
        align(str(input_dir), str(output_dir))
        if args.timer:
            print("Took:", time() - start, "seconds.")

    if args.method[0] == 'm':
        pop, total_acc, mean, std = multiple_pop_eval(gold_standard_dir, output_dir, args.true_tolerance, args.method[1:])
        plot_meval(pop, total_acc, mean, std, args.true_tolerance, args.true_algo)
        return pop, total_acc, mean, std
    else:
        correct_count, total_count = single_eval(gold_standard_dir, output_dir, args.true_tolerance, args.method)
        print(args.true_algo, '[' + str(args.true_tolerance) + '] accuracy:', correct_count/total_count, '[', correct_count, '/', total_count, ']')
        return correct_count, total_count

if __name__ == '__main__':
    this_file = Path(__file__)
    table = ''
    
    parser = ArgumentParser(this_file.name)
    parser.add_argument('data', metavar='data_dir', help='Path to directory containing gold and output annotations')
    parser.add_argument('input', metavar='input_dir', help='Path to input directory')
    parser.add_argument('--algo', '-a', type=str, default='mfa', help='Algorithm to use', choices=['mfa', 'sppas', 'both'])
    parser.add_argument('--tolerance', '-t', nargs='+', type=float, default=0.025, help='Tolerance to determine if the word is correctly aligned')
    parser.add_argument('--method', '-m', type=str, default='std', help='Evaluation method to use', choices=['std', 'outer', 'mstd', 'mouter', 'both'])
    parser.add_argument('--eval', '-e', default=False, help='Only evaluate and skip alignment if output directory already exist', action='store_true')
    parser.add_argument('--save', '-s', type=str, metavar='save', default=None, help='Path where to save the figure')
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    parser.add_argument('--timer', '-ti', default=False, action='store_true')
    args = parser.parse_args()

    input_dir = this_file.parent.joinpath(args.input).resolve()
    data_dir = this_file.parent.joinpath(args.data)

    if len(args.tolerance) > 1 and args.method[0] == 'm':
        args.method = args.method[1:]
        args.tolerance = sorted(args.tolerance)
        print('Cannot use multiple eval while using multiple tolerances.', args.method, 'will be used instead.')

    tolerances = []
    total_acc = {}
    methods = ['std', 'outer'] if args.method == 'both' else [args.method]

    algos = []
    if any(args.algo == a for a in ['sppas', 'both']):
        algos.append(('sppas', align_sppas))

    if any(args.algo == a for a in ['mfa', 'both']):
        algos.append(('mfa', align_mfa))

    for tolerance in args.tolerance:
        args.true_tolerance = tolerance
        tolerances.append(tolerance)

        for true_algo, align in algos:
            args.true_algo = true_algo
            for method in methods:
                args.method = method
                e = eval(align, args)
                if len(e) == 2:
                    correct_count, total_count = e
                    name = args.true_algo + ' [' + method + ']'
                    if name not in total_acc:
                        total_acc[name] = []
                    total_acc[name].append(correct_count / total_count)
                else:
                    pop, acc, mean, std = e
                    name = args.true_algo + ' [' + method[1:] + ']'
                    total_acc['Sam - ' + name] = ['%.3f' % a for a in acc]
                    total_acc['Sub - ' + name] = ['%.3f ± %.4f' % (m, s) for m, s in zip(mean, std)]
    
    if len(tolerances) > 1:
        algos = ['Tolerance (s)']
        accs = []
        for algo, acc in total_acc.items():
            if len(acc) > 0:
                algos.append(algo)
                accs.append(['%.3f' % a for a in acc])
                plot_mtolerance(tolerances, acc, algo)
        table = generate_table('Accuracies of FA models at different tolerances', ['$\leq$%.2f' % t for t in tolerances], algos, list(map(list, zip(*accs))))
    elif args.method[0] == 'm':
        algos = ['Sample size']
        accs = []
        for algo, acc in total_acc.items():
            algos.append(algo)
            accs.append(acc)
        table = generate_table('Accuracies of FA models on population and samples with a tolerance of ' + str(args.tolerance[0]), pop, algos, list(map(list, zip(*accs))))

    if args.verbose:
        print('-' * 20)
        print(table)

    if args.save is not None:
        if '.png' not in args.save:
            args.save += '.png'
        plt.savefig(args.save, bbox_inches='tight')
   
    plt.show()