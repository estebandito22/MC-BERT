#class for tokenizing for the MCB Original
import re

def tokenize(sent):
    t_str = sent.lower()
    for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\.', r'\;']:
        t_str = re.sub(i, '', t_str)
    for i in [r'\-', r'\/', r'\,']:
        t_str = re.sub(i, ' ', t_str)

    return t_str.split()





