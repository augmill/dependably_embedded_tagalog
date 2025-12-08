import pyconll
import json
import os


#dependency data
data = pyconll.load_from_file('tl_ugnayan-ud-test.conllu')

#"triple" h,r,d
#"embeddings"
#"sent_id"
#"neg_sample" BOOL

for sent_id, sentence in enumerate(data):
    #token.form is the name/word itself
    emb = [0.0] * 768

    for token in sentence:
        if '-' in token.id or '.' in token.id:
            continue
        if token.head != '0': #skip root, since it finds the head
            #tuple of head,dependent,relation
            triple = ([sentence[int(token.head)-1].form, sentence[int(token.id)-1].form, token.deprel])

        #make jsons like in Theodoropoulis offline_graph_creation output
        graph_json = {"triple": triple, "embeddings": emb, "sentence_id": sent_id, "neg_sample":False}
    
        with open(os.path.join(".", f"{sent_id}_{int(token.id)-1}.json"), 'w', encoding='utf-8') as file:
            json.dump(graph_json, file, ensure_ascii=False)


    
    