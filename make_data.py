import json
import torch
import pyconll
from tqdm import tqdm

" for train: node_embeddings, adj_m, enc_sent, atte_mask, indexes_of_interest_shifted = batch"
"""
main model reads in a json of splits 
list of file names (split_num.json) passed to ade_daraset in utils 

may be easier to just have our own dataset class 

looks to be set of positive graph and then all of the negative ones for that graph 
"""

def get_triple(sent, t: pyconll.unit.token.Token) -> list:
    """
    Creates the triple [head, dependent, relation] 

    :param ```pyconll.unit.sentence.Sentence``` sent: Sentence object from which to get tokens
    :param ```pyconll.unit.token.Token``` t: Token to look at 

    :returns list: [head, dependent, relation]
    """
    # head = sent[int(str(t.head))-1].form
    # dep = sent[int(t.id)-1].form
    # rel = t.deprel
    # return [head, dep, rel]
    return [sent[int(t.head)-1].form, sent[int(t.id)-1].form, t.deprel]

def make_neg(rs, ks, s_id) -> list:
    """
    Creates negatively sampled graphs that take the dependents from other heads and goes through all of them 
    such that a negatively sampled graph is (dependent, head, relation).

    :param dict rs: Relations where the key is the (head, relation) tuple and the value is a list of the 
    dependents for that head relation pair from a given sentence. 
    :param list ks: All (head, relation) tuples 
    :params str s_id: Relevant sentance I.D.

    :returns list: [{triple:, embedding:, sentence_id:, neg_sample:True}]
    """
    neg_samples = []
    for pair in ks:
        for i in range(len(ks)):
            if ks[i] != pair:
                for dep in rs[pair].keys():
                    # neg_samples.append({"triple": ([ks[i][0], dep, ks[i][1]]), "embeddings": rs[pair][dep], 
                    #     "sentence_id": s_id, "neg_sample":True})
                    neg_samples.append({"triple": ([dep, ks[i][0], ks[i][1]]), "embeddings": rs[pair][dep], 
                        "sentence_id": s_id, "neg_sample":True})
    return neg_samples

def get_embedding():
    """
    Gets embeddings 
    """
    # return torch.zeros(768)
    return [0.0] # * 768

def make_graphs(data) -> dict:
    """
    Creates positive and negative graphs for each sentence

    :param ```pyconll.unit.conll.Conll``` data: All sentences of which to make graphs

    :returns dict: {sentence_id: [positive_samples, negative_samples]}
    """
    # sents = {}
    with open("data/graphs.jsonl", "w") as f:
        for sent_id, sentence in enumerate(tqdm(data)):
            relations = {} 
            keys = []
            emb = get_embedding()
            pos_samples = []
            for token in sentence:
                if '-' not in token.id and '.' not in token.id and token.head != '0':
                    triple = get_triple(sentence, token)
                    ident = (triple[0], triple[2])
                    pos_samples.append({"triple": (triple), "embeddings": emb, "sentence_id": sent_id, 
                                        "neg_sample":False})
                    # neg_graph = {"triple": ([triple[1], triple[0], triple[2]]), "embeddings": emb, 
                    #         "sentence_id": sent_id, "neg_sample":True}
                    if ident not in relations.keys():
                        relations[ident] = {triple[1]: emb}
                        keys.append(ident)
                    elif triple[1] not in relations[ident]: relations[ident][triple[1]] = emb
            neg_samples = make_neg(relations, keys, sent_id)
            json.dump({sent_id : [pos_samples, neg_samples]}, f)
            f.write('\n')

    # sents


"""
make all possible negative by switching relation
finding all possible out comes for a given relation (or head relation pair) type and then assigning 
ones from others to this one that it didn't have 
find out more about the lang and negative sample based on that

head and relation tokens 

loop through all instances, make dict 
{head (token.head-1),relation(token.deprel): [all, possible, deps(token.id)-1]}
check by seeing if a negative sample relation exists in positive sample then deleting from negative 
"""