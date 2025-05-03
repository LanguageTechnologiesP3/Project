import test_utils as tu
import json


def train(dataset_jsonl, n_repeats = 1, train_validate_ratio = 0.75):
    with open(dataset_jsonl, 'r') as dsf:
        dataset = [json.loads(l) for l in dsf]
    
    labeled_sets = tu.label_sort_data(dataset)
    labels = labeled_sets.keys()
    sets = {l : {'train': [], 'validate': []} for l in labels}

    results = []
    
    for _ in range(n_repeats):
        # randomize sets
        for k, v in labeled_sets.values():
            sets[k] = tu.split_dataset(v, {'train': train_validate_ratio, 'validate': 1 - train_validate_ratio})
        
        # train
        model = None
        # validate
        for k, v in sets.values():
            results.append(tu.validate(model, v['validate'], None, labels))
            
        
def main():
    train('Unformatted_Balanced.jsonl')
    

if __name__ == '__main__':
    main()