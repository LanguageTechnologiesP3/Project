import ast
import json
import re
import argparse
from collections import OrderedDict
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

    def compare_to(self, other):
        greater = self.data if len(self.data) > len(other.data) else other.data
        lesser = self.data if len(self.data) <= len(other.data) else other.data
        
        if len(lesser) < 1:
            return 1
        
        diff = 0
        for k, v in lesser.items():
            diff += abs(v - greater[k]) if k in greater else 1

        diff /= len(lesser)
        return diff


class ProfileConstructor:
    def __init__(self, ngram_len, extractor_f: Callable[[list, dict, int], None], preprocess_f: Callable[[any], any] | None = None):
        self.ngrams = {}
        self.extractor_f = extractor_f
        self.preprocess_f = preprocess_f
        self.n = ngram_len

    @staticmethod
    def from_data(import_data: dict, key_deserializer: Callable[[str], any] | None = None):
        if key_deserializer is None:
            data = import_data['data']
        else:
            data = {key_deserializer(k): v for k, v in import_data['data'].items()}
        return Profile(data, import_data['ngram_len'], import_data['count'])


    def add_sequence(self, sequence):
        if self.preprocess_f is not None:
            sequence = self.preprocess_f(sequence)
        for n in range(1, self.n + 1):
            self.extractor_f(sequence, self.ngrams, n)

    def bake_profile(self, n) -> Profile:
        if len(self.ngrams) < 1:
            return Profile({}, self.n, 0)
        
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

        output = {k: v / (rank + 1) for k, v in output.items()}
        return Profile(output, self.n, ngram_count)
        
