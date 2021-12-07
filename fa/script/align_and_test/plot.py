import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TKAgg')

def plot_meval(pop, total_acc, mean, std, tolerance=None, algo=''):
    tmp = ' - ' + algo
    if tolerance:
        tmp += ' [' + str(tolerance) + ']'
    label_pop = 'Sam' + tmp
    label_sam = 'Sub' + tmp

    plt.errorbar(pop, mean, yerr=std, linestyle='None', fmt='.', label=label_sam, alpha=0.4)
    p = plt.plot(pop, total_acc, alpha=0.7)
    plt.scatter(pop, total_acc, c=p[0].get_color(), label=label_pop, s=15, alpha=0.9)
    
    # plt.title('Evolution of accuracy based on number of examples')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Number of examples')
    plt.locator_params(axis='y', nbins=20)
    plt.xticks(pop)
    plt.yticks(np.arange(0.88, 1.01, 0.01))
    
    plt.grid(True, axis='y', ls='--', alpha=0.4)
    plt.legend(loc="lower right")

def generate_table(title, first_column, first_row, values):
    first_column_emphasis = False

    # First column emphasis if necessary
    if len(first_column) == len(values):
        first_column_emphasis = True
        for i, value in enumerate(zip(first_column, values)):
            value[1].insert(0, value[0])

    table = "\\begin{table}\n\\centering\n\\parbox{1.0\\textwidth}{\caption{" + title + "}\\label{tbl:LABELTODO}}\n\\begin{tabular}{"
    # Header
    first = ''
    if len(first_row) < len(values[0]):
        table += '|c'
        first += '& '

    for value in first_row:
        table += '|c'
        first += '{\\bf ' + str(value) + '} & '
    first = first[:-2] + '\\\\\n' # remove last &
    table += '|}\n    \\hline\n        ' + first
        
    for value in values:
        first = ''
        for v in value:
            first += str(v) + ' & '
        first = first[:-2] + '\\\\\n' # remove last &
        table += '    \\hline\n        ' + first
    table += '    \\hline\n\\end{tabular}\n\\end{table}'
    
    return table

def plot_mtolerance(tolerances, total_acc, algo=''):

    p = plt.plot(tolerances, total_acc, alpha=0.4)
    plt.scatter(tolerances, total_acc, c=p[0].get_color(), label=algo, alpha=0.8, s=8)
    
    plt.ylabel('Accuracy')
    plt.xlabel('Tolerance (s)')
    plt.locator_params(axis='y', nbins=20)
    
    plt.grid(True, axis='y', ls='--', alpha=0.4)
    plt.legend(loc='best')