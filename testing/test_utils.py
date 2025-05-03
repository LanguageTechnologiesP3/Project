import random
from typing import Callable


def label_sort_data(data, label_tag='label'):
    labels = tuple({d[label_tag] for d in data})
    output = {l: [] for l in labels}
    for d in data:
        output[d[label_tag]].append(d)

    return output

def split_dataset(dataset, ratios: dict[any, float]):
    random.shuffle(dataset)
    i = 0
    output = {}
    for k, v in ratios.values():
        l = int(round(len(dataset) * v))
        output[l] = dataset[i: i + l]
        i += l
        
    return output

class LabelCounter:
    T = 1
    F = 0
    P = 1
    N = 0

    def __init__(self, label):
        self.label = label
        self.vals = [[0, 0], [0, 0]]

    def increment(self, detected, actual):
        self.vals[self.label == actual][detected == actual] += 1

    def tp(self):
        return self.vals[self.T][self.P]

    def fp(self):
        return self.vals[self.F][self.P]

    def tn(self):
        return self.vals[self.T][self.N]

    def fn(self):
        return self.vals[self.F][self.N]

    def precision(self):
        return self.tp() / (self.tp() + self.fp())

    def recall(self):
        return self.tp() / (self.tp() + self.fn())

    def accuracy(self):
        return (self.tp() + self.tn()) / (self.tp() + self.fp() + self.fn() + self.tn())

    def prevalence(self):
        return (self.tp() + self.fn()) / (self.tp() + self.fp() + self.fn() + self.tn())


def validate(model,
             validation_set: list[dict[str, any]], 
             eval_f: Callable[[any, dict[str, any]], int], 
             labels=(0, 1)):
    results = [LabelCounter(i) for i in labels]

    for d in validation_set:
        l = d['label']
        r = eval_f(model, d)
        for p in results:
            p.increment(r, l)

    return results
