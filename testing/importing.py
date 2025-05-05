import json
import numpy as np


def import_profiling_results(file_path):
    x = []
    y = []
    with open(file_path) as f:
        for line in f:
            d = json.loads(line)
            s = d['scores']
            x.append([s[0]['tokens'], s[0]['names'], s[0]['comments'], s[1]['tokens'], s[1]['names'], s[1]['comments']])
            y.append(d['label'])
    
    return np.array(x), np.array(y)