from evaluation import calculate_metrics, is_correct, METHOD, TOLERANCE

if __name__ == '__main__':
    golds = []
    preds = []
    words = []

    with open('./pipeline/src/results/result', 'r') as f:
        for i, l in enumerate(f):
            arr = [[float(b) for b in a.split("-")] for a in l.replace("\n", '').split(",") if a != '']
            if i % 2 == 0:
                golds.append(arr)
            else:
                preds.append(arr)

    with open('./pipeline/src/results/word_result', 'r') as f:
        for i, l in enumerate(f):
            arr = l.replace("\n", "").split(',')
            words.append(arr)

    tp = 0
    fp = 0
    fn = 0
    a = 0
    b = 0

    for gold, pred, word in zip(golds, preds, words):
        for g, p, w in zip(gold, pred, word):
            if g[0] == 0.0 and g[1] == 0.0:
                print("\tFP &", w, "\\\\\n\\hline")
                fp += 1
            elif p[0] == 0.0 and p[1] == 0.0:
                print("\tFN &", w, "\\\\\n\\hline")
                fn += 1
            elif is_correct(g, p, TOLERANCE, METHOD):
                print("\tTP &", w, "\\\\\n\\hline")
                tp += 1
            else:
                print("\tFN &", w, "\\\\\n\\hline")
                fn += 1

    print('tp:', tp, 'fp:', fp,'fn:', fn)

    print('f1 score: {}, precision: {}, recall: {}'.format(*calculate_metrics(tp, fp, fn)))