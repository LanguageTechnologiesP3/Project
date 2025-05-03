import tokenize
import json
import argparse
import io
import keyword

from six import StringIO


def jsonl_tokenize_code(jsonl_file, output_jsonl, code_tag="code"):
    with open(jsonl_file, "rb") as f:
        with open(output_jsonl, "wb") as of:
            i = 0
            for line in f:
                print("Converting document " + str(i))
                try:
                    converted_json = convert_json(line, code_tag)
                    of.write((json.dumps(converted_json) + "\n").encode("utf-8"))
                except tokenize.TokenError as te:
                    print("FAIL! ")
                    print(te)
                    print(line)
                i += 1


def convert_json(json_str, code_tag="code"):
    json_obj = json.loads(json_str)
    code = json_obj[code_tag].replace("$\\n", "\n")
    reader = io.BytesIO(code.encode("utf-8"))
    tokens = tokenize.tokenize(reader.readline)
    tok_list = []
    for tok in tokens:
        tok_id = tok.exact_type
        if tok_id == tokenize.NAME and keyword.iskeyword(tok.string):
            tok_id = keyword.kwlist.index(tok.string) + 70
        elif tok_id == tokenize.SOFT_KEYWORD and keyword.issoftkeyword(tok.string):
            tok_id = keyword.softkwlist.index(tok.string) + 70 + len(keyword.kwlist)
        tok_list.append(tok_id)

    # del json_obj[code_tag]
    json_obj["tokenized_code"] = tok_list
    return json_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    jsonl_tokenize_code(args.file, args.output)
