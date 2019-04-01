import csv
import sys
from collections import Counter

from util.mcbtokenizer import MCBTokenizer

'''
This script will take in a list of csv files where the second field is a text string, and build a dictionary.
'''

if len(sys.argv) == 1:
    print("Usage:", sys.argv[0], "glove.txt questions1.csv questions2.csv ...")
    exit(1)

all_words = []

tokenizer = MCBTokenizer(None) #don't actually have a dict yet

for cfile in sys.argv[2:]:
    print("parsing", cfile)
    with open(cfile) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            all_words.extend(tokenizer.tokenize(row[1]))

token_counter = Counter(all_words)
vocab, count = zip(*token_counter.most_common(1_000_000))
id2token = list(vocab)

print("inital vocab size:", len(id2token))

swords = set(id2token)

glove_lookup = {}

#lookup all our words and cache the flove encoding
with open(sys.argv[1]) as f:
    for l in f:
        splt = l.split()
        w = splt[0]
        if w in swords:
            glove_lookup[w] = splt[1:]
            swords.remove(w)

#remove the words we couldn't find an encoding for
print("unfound words:", swords)

for w in swords:
    id2token.remove(w)


print("vocab of", len(id2token))

with open('dict.txt', 'w') as f:
    for tok in id2token:
        f.write(tok + ' ' + ' '.join(glove_lookup[tok]) + '\n')

