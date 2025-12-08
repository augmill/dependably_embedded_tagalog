import pyconll
import json
from graph_creation.make_data import make_graphs
# NOTE: run from 

# make positive graph list 
pos = []
with open("../data/graphs.jsonl", "r") as f:
    for line in f: 
        for key, value in json.loads(line).items():
            for sample in value[0]:
                pos.append(sample["triple"])

# make new file with the checked graphs 

with open("../data/graphs.jsonl", "r") as in_f:
    with open("../data/checked_graphs.jsonl", "w") as out_f:
        for i, line in enumerate(in_f): 
            good = []
            for key, value in json.loads(line).items():
                for sample in value[1]:
                    if sample["triple"] not in pos: good.append(sample)
            json.dump({key : [value[0], good]}, out_f)
            out_f.write('\n')