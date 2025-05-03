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
    names = set()
    comments = []
    for tok in tokens:
        tok_id = tok.exact_type
        if tok_id == tokenize.NAME:
            if keyword.iskeyword(tok.string):
                tok_id = keyword.kwlist.index(tok.string) + 70
            else:
                names.add(tok.string)
        elif tok_id == tokenize.SOFT_KEYWORD and keyword.issoftkeyword(tok.string):
            tok_id = keyword.softkwlist.index(tok.string) + 70 + len(keyword.kwlist)
        elif tok_id == tokenize.COMMENT:
            comments.append(tok.string.removeprefix('#').strip())
            
        tok_list.append(tok_id)

    output = {
        "label": json_obj["label"],
        "tokens": tok_list,
        "names": list(names),
        "comments": comments,
    }

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    jsonl_tokenize_code(args.file, args.output)
