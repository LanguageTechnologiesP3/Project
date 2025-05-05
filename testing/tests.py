import test_utils as tu
import DNN as dnn
import json
import numpy as np


def train(dataset_jsonl, n_repeats = 1, train_validate_ratio = 0.75):
    model =  dnn.build_DNN(input_file=dataset_jsonl, test_split=train_validate_ratio)
    
    
    with open(dataset_jsonl, 'r') as dsf:
        dataset = [json.loads(l) for l in dsf]
    
    labeled_sets = tu.label_sort_data(dataset)
    labels = labeled_sets.keys()
    sets = {l : {'train': [], 'validate': []} for l in labels}

    results = []
    
    for _ in range(n_repeats):
        # randomize sets
        for k, v in labeled_sets.items():
            sets[k] = tu.split_dataset(v, {'train': train_validate_ratio, 'validate': 1 - train_validate_ratio})
        
        # train
        #model =  dnn.build_DNN(input_file=dataset_jsonl)
        
        
        # validate
        for k, v in sets.values():
            results.append(tu.validate(model, v['validate'], None, labels))
            
        
def main():
    train('./src/res.jsonl')
    

if __name__ == '__main__':
    main()