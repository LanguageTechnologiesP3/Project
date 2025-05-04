import ast
import json
import re
import argparse
from collections import defaultdict, OrderedDict
from typing import Callable
from ngram_profile import FullProfiler


def read_jsonl(jsonl_file, output_jsonl, settings):  
    profiler = FullProfiler(settings)
    profiler.init_constructors()

    with open(jsonl_file, "rb") as f:
        for line in f:
            item = json.loads(line)
            profiler.add_example(item)

    profiler.bake_profiles()
    
    with open(output_jsonl, "w") as f:
        f.write(profiler.to_json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--output")
    parser.add_argument("-n", "--number", type=int)
    parser.add_argument("-m", "--max", type=int)
    parser.add_argument("-p", "--preprocess", type=bool)
    args = parser.parse_args()

    read_jsonl(args.file, args.output, {
        "tokens":{
            "ngram_len": args.max,
            "max_ngrams": args.number
        },
        "names":{
            "ngram_len": args.max,
            "max_ngrams": args.number,
            "preprocess": args.preprocess
        },
        "comments":{
            "ngram_len": args.max,
            "max_ngrams": args.number,
            "preprocess": args.preprocess
        }
    })