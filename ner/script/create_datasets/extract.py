from glob import iglob, glob
from variables import DETERMINANTS, LABELS_TO_SKIP, LABELS_TO_CHANGE

def parse(line: str):
    label, start, end = line.split('\t')[1].split(' ')
    return int(start), int(end), label

def extractDatasets(pathToDir: str, verbose: int):
    examples = []
    
    # Loop over all directories
    for path in glob(pathToDir + '/*'):
        
        # Loop over all documents in directory
        files = list(iglob(path + '/*.ann'))

        for ann in files:
            if verbose >= 2:
                print(ann)
            
            # Read text
            txt = ann[:-3] + 'txt'
            with open(txt, 'r', encoding='utf-8') as f:
                text = f.read().lower() # Lowercase everything

            for c in ['\n', '\t', '\r', '\f', '\a']:
                text = text.replace(c, ' ')

            text = text.replace('\xa0', ' ').replace('\u2009', ' ').replace('\ufeff', ' ')
            
            # Read annotations
            annotations = []
            with open(ann, 'r') as f:
                for line in f.readlines():
                    if line[0] != 'T':
                        continue
                    start, end, label = parse(line)
                    if label in LABELS_TO_SKIP:
                        if verbose >= 1:
                            print('SKIPPING LABEL', label)
                    else:
                        if label in LABELS_TO_CHANGE:
                            if verbose >= 1:
                                print('CHANGING', label, 'to', LABELS_TO_CHANGE[label])
                            label = LABELS_TO_CHANGE[label]
                        annotations.append((start, end, label, line))
                    
            annotations.sort(key=lambda x: x[0])
            
            # Add spaces where needed so spacy doesnt return none when creating spans
            ents = []
            jump = 0
            previous_end = 0

            # Remove leading spaces in text
            while text[0] == ' ':
                text = text[1:]
                jump -= 1

            for start, end, label, line in annotations:
                start += jump
                end += jump

                # Remove leading and ending spaces from entity
                while text[start] == ' ':
                    start += 1

                while text[end-1] == ' ':
                    end -= 1

                # Remove double spaces between this entity and the previous one
                while '  ' in text[previous_end:start]:
                    text = text[:previous_end] + text[previous_end:start].replace('  ', ' ', 1) + text[start:]
                    jump -= 1
                    start -= 1
                    end -= 1

                # Remove leading determinant
                for j in range(2, 5):
                    if text[start:start+j] in DETERMINANTS[j-2]:
                        if verbose >= 1:
                            print("REMOVING DET", text[start:start+j], "from the entity")
                        start += j
                
                # Add spaces at the start so spacy can split entities correctly
                if start > 0 and text[start-1] in 'abcdefghijklmnopqrstuvwxyz,0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ.':
                    if verbose >= 2:
                        print('-'*20, '\n', line[:-1])
                        print('START', label, text[start-1:end])
                    jump += 1
                    text = text[:start] + ' ' + text[start:]
                    start += 1
                    end += 1
                    
                    if verbose >= 2:
                        print('START2', label, text[start-1:end])
                
                # Add spaces at the end so spacy can split entities correctly
                if end < len(text) and text[end] in 'abcdefghijklmnopqrstuvwxyz()[]-ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    if verbose >= 2:
                        print('-'*20, '\n', line[:-1])
                        print('END', label, text[start:end+1])
                    jump += 1
                    text = text[:end] + ' ' + text[end:]
                    
                    if verbose >= 2:
                        print('END2', label, text[start:end+1])

                # Remove double spaces in entity
                while '  ' in text[start:end]:
                    text = text[:start] + text[start:end].replace('  ', ' ', 1) + text[end:]
                    jump -= 1
                    end -= 1

                # print("-----")
                # print(previous_end, start, end, '"' + text[previous_end:start] + '"', '"' + text[start:end] + '"')
                
                previous_end = end
                ents.append((start, end, label, line))

            # Remove double spaces between last entity and end of text
            while '  ' in text[previous_end:]:
                text = text[:previous_end] + text[previous_end:].replace('  ', ' ', 1)

            examples.append((text, ents, ann))

    return examples