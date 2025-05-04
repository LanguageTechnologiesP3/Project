import ast
import json
import argparse
from ngram_profile import Profile, ProfileConstructor, extract_char_ngrams, extract_number_ngrams, make_preprocess_f


def load_profiles_from_file(file_path: str) -> tuple[dict, list[dict[str, Profile]]]:
    with open(file_path, "rb") as f:
        data = json.load(f)
    
    settings = data['settings']
    profiles = []
    for profile_data in data['profiles']:
        profiles_tmp = {}
        for key, value in profile_data.items():
            if key == 'tokens':
                profiles_tmp[key] = ProfileConstructor.from_data(value, ast.literal_eval)
            else:
                profiles_tmp[key] = ProfileConstructor.from_data(value)
        profiles.append(profiles_tmp)
    return settings, profiles


def classify(models_json, dataset_jsonl, output_jsonl):
    
    settings, profiles = load_profiles_from_file(models_json)
    char_preprocessor = make_preprocess_f(settings['preprocess'])
    

    with open(dataset_jsonl, "rb") as f:
        with open(output_jsonl, "w") as output:
            for line in f:
                item = json.loads(line)
    
                constructors = {
                    "tokens": ProfileConstructor(settings['ngram_len'], extract_number_ngrams),
                    "names": ProfileConstructor(settings['ngram_len'], extract_char_ngrams, char_preprocessor),
                    "comments": ProfileConstructor(settings['ngram_len'], extract_char_ngrams, char_preprocessor),
                }
                
                label = item.get("label")
                for k, v in constructors.items():
                    v.add_sequence(item.get(k))
                
                scores = [{k: v.compare_to(constructors[k].bake_profile(settings["max_ngrams"])) for k, v in p.items()} for p in profiles]
                    
                res = {
                    "label": label,
                    "votes": {k: 0 if scores[0][k] <= scores[1][k] else 1 for k in constructors.keys()},
                    "scores": scores
                }
                
                output.write(json.dumps(res) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-d", "--dataset", required=True)
    args = parser.parse_args()

    classify(args.model, args.dataset, args.output)
