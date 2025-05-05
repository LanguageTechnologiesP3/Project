import argparse
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, default="classified.jsonl")
    args = parser.parse_args()
    
    tokens = [[],[]]
    names = [[],[]]
    comments = [[],[]]
    
    with open(args.data_file, 'r') as f:
        for line in f:
            j = json.loads(line)
            scores = j['scores']
            label = j['label']
            t = tokens[label]
            n = names[label]
            c = comments[label]
            
            t.append(scores[0]['tokens'] - scores[1]['tokens'])
            n.append(scores[0]['names'] - scores[1]['names'])
            c.append(scores[0]['comments'] - scores[1]['comments'])
    
    ax = plt.axes(projection='3d')
    
    ax.scatter(tokens[0], names[0], comments[0])
    ax.scatter(tokens[1], names[1], comments[1])
    
    plt.show()
    
    
if __name__ == "__main__":
    main()