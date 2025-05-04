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
    def __init__(self, ngram_len, extractor_f: Callable[[list, dict, int], None],
                 preprocess_f: Callable[[any], any] | None = None):
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
            ngram_count += v
            if v != last_count:
                rank += 1
                last_count = v
            output[k] = rank
            cnt += 1
            if cnt >= n:
                break

        output = {k: v / (rank + 1) for k, v in output.items()}
        return Profile(output, self.n, ngram_count)


class FullProfiler:
    def __init__(self, settings: dict | None = None):
        self.settings = {
            "tokens": {
                "ngram_len": 5,
                "max_ngrams": 300,
            },
            "names": {
                "ngram_len": 5,
                "max_ngrams": 300,
                "preprocess": False
            },
            "comments": {
                "ngram_len": 5,
                "max_ngrams": 300,
                "preprocess": False
            },
        }
        if settings is not None:
            self.settings.update(settings)

        self.profile_constructors = None
        self.profiles = None

    def _init_constructor_set(self) -> dict:
        names_preprocessor = make_preprocess_f(self.settings['names']['preprocess'])
        comments_preprocessor = make_preprocess_f(self.settings['comments']['preprocess'])
        return {
            "tokens": ProfileConstructor(self.settings['tokens']['ngram_len'], extract_number_ngrams),
            "names": ProfileConstructor(self.settings['names']['ngram_len'], extract_char_ngrams,
                                        names_preprocessor),
            "comments": ProfileConstructor(self.settings['comments']['ngram_len'], extract_char_ngrams,
                                           comments_preprocessor)
        }

    def init_constructors(self):
        self.profile_constructors = [
            self._init_constructor_set(),
            self._init_constructor_set()
        ]

    def add_example(self, example: dict):
        label = example.get("label")
        for k, v in self.profile_constructors[label].items():
            v.add_sequence(example.get(k))

    def bake_profiles(self):
        self.profiles = [{k: v.bake_profile(self.settings[k]['max_ngrams']) for k, v in constructors.items()}
                         for constructors in self.profile_constructors]

    def to_json(self):
        return json.dumps({
            "settings": self.settings,
            "profiles": [{k: v.__dict__() for k, v in p.items()} for p in self.profiles]
        })

    def from_json(self, model_json):
        with open(model_json, "r") as f:
            data = json.load(f)

        self.settings = data['settings']
        self.profiles = []
        for profile_data in data['profiles']:
            profiles_tmp = {}
            for key, value in profile_data.items():
                if key == 'tokens':
                    profiles_tmp[key] = ProfileConstructor.from_data(value, ast.literal_eval)
                else:
                    profiles_tmp[key] = ProfileConstructor.from_data(value)
            self.profiles.append(profiles_tmp)

    def compare(self, data_point):
        constructors = self._init_constructor_set()

        label = data_point.get("label")
        for k, v in constructors.items():
            v.add_sequence(data_point.get(k))

        scores = [
            {k: v.compare_to(constructors[k].bake_profile(self.settings[k]["max_ngrams"])) for k, v in p.items()} for
            p in self.profiles]

        return {
            "label": label,
            "votes": {k: 0 if scores[0][k] <= scores[1][k] else 1 for k in constructors.keys()},
            "scores": scores
        }
