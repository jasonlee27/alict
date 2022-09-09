# This script extracts pos/neg/neu words
# based on the score computed in SentiWordNet
# Link: https://github.com/aesuli/SentiWordNet
# The label is assigned from the algorithm introduced at the paper
# Mihaela Colhon et al.
# Paper Link: https://www.mdpi.com/2073-8994/9/11/280/htm

from typing import *

import re, os
import sys
import json
import random

from pathlib import Path
from ...utils.Macros import Macros
from ...utils.Utils import Utils

class Sentiwordnet:

    @classmethod
    def read_raw_data(cls):
        data = dict()
        lns = Utils.read_txt(Macros.swn_data_file)
        for l in lns:
            if not l.strip().startswith("#"):
                l_split = l.split("\t")
                word_key_list = l_split[4].split()
                if len(word_key_list)==1:
                    word_key = word_key_list[0].split("#")[0]+"::"+l_split[0]
                    if word_key not in data.keys():
                        data[word_key] = list()
                    # end if
                    data[word_key].append({
                        "id": l_split[1],
                        "pos_score": l_split[2],
                        "neg_score": l_split[3],
                        "synsetterms": l_split[4],
                        "order": int(l_split[4].split("#")[-1])
                    })
                else:
                    for word_key in word_key_list:
                        _word_key = word_key.split("#")[0]+"::"+l_split[0]
                        if _word_key not in data.keys():
                            data[_word_key] = list()
                        # end if
                        data[_word_key].append({
                            "id": l_split[1],
                            "pos_score": l_split[2],
                            "neg_score": l_split[3],
                            "synsetterms": l_split[4],
                            "order": int(l_split[4].split("#")[-1])
                        })
                    # end for
                # end if
            # end if
        # end for
        for w in data.keys():
            data[w] = sorted(data[w], key=lambda x: x["order"])
        # end for
        return data

    @classmethod
    def get_word_score(cls, raw_data):
        data = None
        if os.path.exists(Macros.result_dir / "swn_word_scores.json"):
            return Utils.read_json(Macros.result_dir / "swn_word_scores.json")
        else:
            data = dict()
            for word, scores in raw_data.items():
                pos_scores = [float(s["pos_score"]) for s in scores]
                neg_scores = [float(s["neg_score"]) for s in scores]
                denom = sum([1./i for i in range(1,len(pos_scores)+1)])
                agg_pos_score = sum([(1./i)*(pos_scores[i-1]**i) for i in range(1,len(pos_scores)+1)])/denom
                agg_neg_score = sum([(1./i)*(neg_scores[i-1]**i) for i in range(1,len(neg_scores)+1)])/denom
                obj_score = 1-agg_pos_score-agg_neg_score
                data[word] = {
                    "agg_pos_score": agg_pos_score,
                    "agg_neg_score": agg_neg_score,
                    "obj_score": obj_score,
                    "pos_scores": pos_scores,
                    "neg_scores": neg_scores
                }
            # end for
            
            # Write score data
            Utils.write_json(data, Macros.result_dir / "swn_word_scores.json", pretty_format=True)
        # end if
        return data
        
    
    @classmethod
    def get_word_label(cls, data):
        def get_pos(tag):
            """
            Convert between the PennTreebank tags to simple Wordnet tags
            """
            if tag.startswith('a'):
                return "adj"
            elif tag.startswith('n'):
                return "noun"
            elif tag.startswith('r'):
                return "adj"
            elif tag.startswith('v'):
                return "verb"
            return None
        result = dict()
        for d in data.keys():
            synset_scores = [ps-ns for ps,ns in zip(data[d]["pos_scores"],data[d]["neg_scores"])]
            score = sum([(1./i)*(synset_scores[i-1]**i) for i in range(1,len(synset_scores)+1)])
            label = "neutral"
            if score>=0.75:
                label = "positive" # strong positive
            elif score>0.25 and score<=0.75:
                label = "positive"
            elif score>0. and score<=0.25:
                label = "positive" # weak positive
            elif score<0. and score>=-0.25:
                label = "negative" # weak negative
            elif score<-0.25 and score>=-0.75:
                label = "negative" # negative
            elif score<=-0.75:
                label = "negative"
            # end if
            if label=="neutral":
                if data[d]["agg_pos_score"]==0 and data[d]["agg_neg_score"]==0 and data[d]["obj_score"]==1:
                    label = "pure neutral"
                elif data[d]["agg_pos_score"]>=0.006 and data[d]["agg_pos_score"]<=0.41 and \
                     data[d]["agg_neg_score"]>=0.006 and data[d]["obj_score"]<=0.41 and \
                     data[d]["obj_score"]>=0.16 and data[d]["obj_score"]<=0.98:
                    label = "balanced neutral"
                elif data[d]["agg_pos_score"]==0.5 and data[d]["agg_neg_score"]==0.5 and data[d]["obj_score"]==0:
                    label = "half-pos-neg neutral"
                # end if
            # end if
            word = d.split("::")[0]
            pos = d.split("::")[-1]
            result[d] = {
                "word": word,
                "POS": get_pos(pos),
                "label": label
            }
        # end for
        # Write label data
        Utils.write_json(result, Macros.result_dir / "swn_word_labels.json", pretty_format=True)
        return data

    @classmethod
    def get_sent_words(cls):
        if os.path.exists(Macros.result_dir / "swn_word_labels.json"):
            data = Utils.read_json(Macros.result_dir / "swn_word_labels.json")
        else:
            rdata = cls.read_raw_data()
            score_data = cls.get_word_score(rdata)
            data = cls.get_word_label(score_data)
        # end if
        return data


# if __name__=="__main__":
#     Sentiwordnet.get_sent_words()
