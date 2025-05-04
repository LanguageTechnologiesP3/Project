import ast
import json
import re
import argparse
from collections import defaultdict, OrderedDict
from typing import Callable
from ngram_profile import Profile, ProfileConstructor, extract_char_ngrams, extract_number_ngrams, make_preprocess_f
from src.ngram_profile import make_preprocess_f, ProfileConstructor, extract_number_ngrams, extract_char_ngrams


def read_jsonl(jsonl_file, output_jsonl, number_of_ngrams, max_n_grams, should_preprocess):  
    char_preprocessor = make_preprocess_f(should_preprocess)
    
    profile_constructors = [
        {
            "tokens": ProfileConstructor(max_n_grams, extract_number_ngrams),
            "names": ProfileConstructor(max_n_grams, extract_char_ngrams, char_preprocessor),
            "comments": ProfileConstructor(max_n_grams, extract_char_ngrams, char_preprocessor),
        },
        {
            "tokens": ProfileConstructor(max_n_grams, extract_number_ngrams),
            "names": ProfileConstructor(max_n_grams, extract_char_ngrams, char_preprocessor),
            "comments": ProfileConstructor(max_n_grams, extract_char_ngrams, char_preprocessor)
        }
    ]

    with open(jsonl_file, "rb") as f:
        for line in f:
            item = json.loads(line)

            label = item.get("label")
            for k, v in profile_constructors[label].items():
                v.add_sequence(item.get(k))

    profiles = [{k: v.bake_profile(number_of_ngrams).__dict__() for k, v in constructors.items()} for constructors in profile_constructors]
    
    with open(output_jsonl, "w") as f:
        
        json.dump({
            "settings": {
                "max_ngrams": number_of_ngrams,
                "ngram_len": max_n_grams,
                "preprocess": should_preprocess
            },
            "profiles": profiles,
        }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--output")
    parser.add_argument("-n", "--number", type=int)
    parser.add_argument("-m", "--max", type=int)
    parser.add_argument("-p", "--preprocess", type=bool)
    args = parser.parse_args()

    read_jsonl(args.file, args.output, args.number, args.max, args.preprocess)