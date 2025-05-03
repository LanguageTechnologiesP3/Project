import json
import re
import argparse
from collections import defaultdict, OrderedDict
from typing import Callable

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
_RE_REMOVE_NON_CHAR = re.compile(r"[^a-zA-Z'\s]")

def preprocessing(input_list, should_save_only_char):
    words = []
    
    for it in input_list:
        item = it.lower()
        if should_save_only_char:
            item = _RE_REMOVE_NON_CHAR.sub('', item)
        
        if ' ' in it:
            item = _RE_COMBINE_WHITESPACE.sub(" ", item).strip()
            words.extend(item.split(' '))
        else:
            words.append(item.strip())
    return words

def make_preprocess_f(should_save_only_char):
    def f(input_list):
        return preprocessing(input_list, should_save_only_char)
    return f


def extract_number_ngrams(input_list, input_dict, n):
    if n < 1 or not input_list:
        return

    padded_start = input_list[0]
    padded_end = input_list[-1]
    padded = input_list.copy()

    if n > 1:
        padded = [padded_start] + padded + [padded_end] * (n - 1)

    for i in range(len(padded) - n + 1):
        ngram = tuple(padded[i:i + n])
        if ngram in input_dict:
            input_dict[ngram] += 1
        else:
            input_dict[ngram] = 1
    

def extract_char_ngrams(input_list, input_dict, n):
    if n < 1 or not input_list:
        return

    padded_start_end = ' '

    for word in input_list:
        padded = padded_start_end + word + padded_start_end * (n - 1)

        for i in range(len(padded) - n + 1):
            ngram = ''.join(padded[i:i + n])
            if ngram in input_dict:
                input_dict[ngram] += 1
            else:
                input_dict[ngram] = 1
        
def sort_dict_by_value_and_return_n_most(input_dict, n=300):
    sorted_dict = OrderedDict(sorted(input_dict.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_dict.keys())[:n]


class Profile:
    def __init__(self, data: dict, ngram_len, count):
        self.data = data
        self.ngram_len = ngram_len
        self.count = count
        
    def __dict__(self):
        data = {str(k): v for k, v in self.data.items()}
        
        return {
            "ngram_len": self.ngram_len,
            "data": data,
            "count": self.count,
        }
    
    def to_json(self):
        return json.dumps(self.data)
    
    def compare_to(self, other):
        greater = self.data if len(self.data) > len(other.data) else other.data
        lesser = self.data if len(self.data) <= len(other.data) else other.data
        
        diff = 0
        for k, v in lesser.items():
           diff += abs(v - greater[k]) if k in greater else 1
        
        diff /= len(lesser)
        return diff     
    
    def load_data(self, profile_data: dict):
        self.data = profile_data["data"]
        self.ngram_len = profile_data["ngram_len"]
        self.count = profile_data["count"]         
        

class ProfileConstructor:
    def __init__(self, ngram_len, extractor_f: Callable[[list, dict, int], None], preprocess_f: Callable[[any], any] | None = None):
        self.ngrams = {}
        self.extractor_f = extractor_f
        self.preprocess_f = preprocess_f
        self.n = ngram_len
        
    def load_data(self, profile_dict: dict):
        self.ngrams = profile_dict.get("data", {})
        self.ngram_len = profile_dict.get("ngram_len")
        self.count = profile_dict.get("count", 0)

        
    def add_sequence(self, sequence):
        if self.preprocess_f is not None:
            sequence = self.preprocess_f(sequence)
        for n in range(1, self.n + 1):
            self.extractor_f(sequence, self.ngrams, n)

    def bake_profile(self, n) -> Profile:
        sorted_dict = OrderedDict(sorted(self.ngrams.items(), key=lambda item: item[1], reverse=True))
        last_count = max(sorted_dict.values())
        
        ngram_count = 0
        output = {}
        rank = 0
        cnt = 0
        for k, v in sorted_dict.items():
            ngram_count += 1
            if v != last_count:
                rank += 1
                last_count = v
            output[k] = rank
            cnt += 1
            if cnt >= n:
                break
        
        output = {k: v / rank for k, v in output.items()}
        return Profile(output, self.n, ngram_count)
        
        
def compare_and_return_class(models_profile_list, test_profile):
    res = []
    for model_profile in models_profile_list:
        diff = 0.0
        count = 0
        for k,v in model_profile.items():
            diff += v.compare_to(test_profile[k])
            count += 1
        res.append(diff/count)
    for it in res:
        print(it)


def load_profiles_from_file(file_path: str) -> dict[str, ProfileConstructor]:
    with open(file_path, "rb") as f:
        data = json.load(f)

    profiles = []

    for profile_data in data:
        profiles_tmp = {}
        for key, value in profile_data.items():
            profile = Profile({}, value["ngram_len"], value["count"])
            profile.load_data(value)
            profiles_tmp[key] = profile
        profiles.append(profiles_tmp)
    return profiles
    
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
        json.dump(profiles, f)


def classify(models_jsonl_file, example_jsonl_file, number_of_ngrams, max_n_grams, should_preprocess):
    char_preprocessor = make_preprocess_f(should_preprocess)
    
    profiles = load_profiles_from_file(models_jsonl_file)
    print(profiles[1]["tokens"].count)
    
    profile_code_constructor = {
        "tokens": ProfileConstructor(max_n_grams, extract_number_ngrams),
        "names": ProfileConstructor(max_n_grams, extract_char_ngrams, char_preprocessor),
        "comments": ProfileConstructor(max_n_grams, extract_char_ngrams, char_preprocessor),
    }
    with open(example_jsonl_file, "rb") as f:
        for line in f:
            item = json.loads(line)
            for k, v in profile_code_constructor.items():
                v.add_sequence(item.get(k))
                
    profile_code = {k: v.bake_profile(number_of_ngrams).__dict__() for k, v in profile_code_constructor.items()}
    
    #compare
    compare_and_return_class(profiles, profile_code)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--output")
    parser.add_argument("-n", "--number", type=int)
    parser.add_argument("-m", "--max", type=int)
    parser.add_argument("-p", "--preprocess", type=bool)
    args = parser.parse_args()

    classify(args.file, args.output, args.number, args.max, args.preprocess)
    #read_jsonl(args.file, args.output, args.number, args.max, args.preprocess)