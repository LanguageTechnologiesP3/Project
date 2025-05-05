import ast
import json
import argparse
from ngram_profile import FullProfiler


def classify(models_json, dataset_jsonl, output_jsonl):
    
    profiler = FullProfiler()
    profiler.from_json(models_json)
    
    with open(dataset_jsonl, "rb") as f:
        with open(output_jsonl, "w") as output:
            for line in f:
                item = json.loads(line)
                res = profiler.compare(item)
                output.write(json.dumps(res) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-d", "--dataset", required=True)
    args = parser.parse_args()

    classify(args.model, args.dataset, args.output)
