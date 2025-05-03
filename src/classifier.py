import json
import re
import argparse
from collections import defaultdict, OrderedDict

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
_RE_REMOVE_NON_CHAR = re.compile(r"[^a-zA-Z'\s]")

def read_jsonl(jsonl_file, output_jsonl, number_of_ngrams, max_n_grams, should_preprocces):
    data = []
    ngram_counts_tokens_human = defaultdict(int)
    ngram_counts_names_human = defaultdict(int)
    ngram_counts_comments_human = defaultdict(int)
    ngram_counts_tokens = defaultdict(int)
    ngram_counts_names = defaultdict(int)
    ngram_counts_comments = defaultdict(int)
    #jsonl_file = "C:/Project/src/res.jsonl"
    #number_of_ngrams = 300
    #max_n_grams = 5
    #should_preprocces = False
    
    with open(jsonl_file, "rb") as f:
        for line in f:
            item = json.loads(line)
            
            tokens = item.get("tokens")
            names = item.get("names")
            comments = item.get("comments")
            
            names = preprocessing(names, should_preprocces)
            comments = preprocessing(comments, should_preprocces)
            
            label = item.get("label")
            if(label == 1):
                for n in range(1,max_n_grams+1):
                    extract_number_ngrams(tokens, ngram_counts_tokens, n)
                    extract_char_ngrams(names, ngram_counts_names, n)
                    extract_char_ngrams(comments, ngram_counts_comments, n)
            else:
                for n in range(1,max_n_grams+1):
                    extract_number_ngrams(tokens, ngram_counts_tokens_human, n)
                    extract_char_ngrams(names, ngram_counts_names_human, n)
                    extract_char_ngrams(comments, ngram_counts_comments_human, n)
            
            
    ngram_dicts = [
        (sort_dict_by_value_and_return_n_most(ngram_counts_tokens, number_of_ngrams), "tokens_ai"),
        (sort_dict_by_value_and_return_n_most(ngram_counts_names, number_of_ngrams), "names_ai"),
        (sort_dict_by_value_and_return_n_most(ngram_counts_comments, number_of_ngrams), "comments_ai"),
        (sort_dict_by_value_and_return_n_most(ngram_counts_tokens_human, number_of_ngrams), "tokens_human"),
        (sort_dict_by_value_and_return_n_most(ngram_counts_names_human, number_of_ngrams), "names_human"),
        (sort_dict_by_value_and_return_n_most(ngram_counts_comments_human, number_of_ngrams), "comments_human")
    ]
    with open(output_jsonl, "w") as f:
        for ngrams, category in ngram_dicts:
            result = {category: ngrams}
            f.write(json.dumps(result))


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
        input_dict[ngram] += 1
        
def extract_char_ngrams(input_list, input_dict, n):
    if n < 1 or not input_list:
        return

    padded_start_end = ' '

    for word in input_list:
        padded = padded_start_end + word + padded_start_end * (n - 1)

        for i in range(len(padded) - n + 1):
            ngram = ''.join(padded[i:i + n])
            input_dict[ngram] += 1
            
def sort_dict_by_value_and_return_n_most(input_dict, n=300):
    sorted_dict = OrderedDict(sorted(input_dict.items(), key=lambda item: item[1], reverse=True))
    #return list(sorted_dict.keys())[:n]
    return list(sorted_dict.items())[:n]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--output")
    parser.add_argument("-n", "--number", type=int)
    parser.add_argument("-m", "--max", type=int)
    parser.add_argument("-p", "--preprocess", type=bool)
    args = parser.parse_args()

    read_jsonl(args.file, args.output, args.number, args.max, args.preprocess)