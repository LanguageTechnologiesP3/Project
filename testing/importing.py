import json
import numpy as np


def import_profiling_results(file_path):
    x = []
    y = []
    with open(file_path) as f:
        for line in f:
            d = json.loads(line)
            x.append([d[0]['tokens'], d[0]['names'], d[0]['comments'], d[1]['tokens'], d[1]['names'], d[1]['comments']])
            y.append(d['label'])
    
    return np.array(x), np.array(y)