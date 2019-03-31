import csv
import sys
from collections import Counter

import util.mcbtokenizer as tokenizer

'''
This script will take in a list of csv files where the second field is a text string, and build a dictionary.
'''

if len(sys.argv) == 1:
    print("Usage:", sys.argv[0], "glove.txt questions1.csv questions2.csv ...")
    exit(1)

all_words = []

for cfile in sys.argv[1:]:
    print("parsing", cfile)
    with open(cfile) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            all_words.extend(tokenizer.tokenize(row[1]))

swords = set(all_words)

glove_lookup = {}

#lookup all our words and cache the flove encoding
with open("glove.txt") as f:
    for l in f:
        splt = l.split()
        w = splt[0]
        if w in swords:
            glove_lookup[w] = splt[1:]
        else:
            swords.remove(w)

#remove the words we couldn't find an encoding for
print("unfound words:", swords)

for w in swords:
    all_words.remove(w)

token_counter = Counter(all_words)
vocab, count = zip(*token_counter.most_common(1_000_000))
id2token = list(vocab)
token2id = dict(zip(vocab, range(len(vocab))))

print("vocab of", len(id2token))

with open('dict.txt', 'w') as f:
    for tok in id2token:
        f.write(tok + ' ' + ' '.join(glove_lookup[tok]) + '\n')

