"""Demonstrates how to make a simple call to the Natural Language API."""
# Score of the sentiment ranges between -1.0 (negative) and 1.0 (positive)
# and corresponds to the overall emotional leaning of the text.
# 
# Magnitude indicates the overall strength of emotion (both positive and negative)
# within the given text, between 0.0 and +inf. Unlike score, magnitude is not normalized;
# each expression of emotion within the text (both positive and negative)
# contributes to the text's magnitude (so longer text blocks may have greater magnitudes).

# import argparse

# from typing import *
# from pathlib import Path

# import numpy as np

# from google.cloud import language_v1

# from ..utils.Macros import Macros

# class GoogleModel:

#     @classmethod
#     def load_model_client(cls):
#         return language_v1.LanguageServiceClient()
        
#     @classmethod
#     def get_batch(cls, l, n):
#         """Yield successive n-sized chunks from l."""
#         for i in range(0, len(l), n):
#             yield l[i:i + n]
#         # end for

#     @classmethod
#     def batch_predict(cls, data: List[str], batch_size: int = 32):
#         preds = list()
#         client = cls.load_model_client()
#         for batch in cls.get_batch(data, batch_size):
#             for d in batch:
#                 document = language_v1.Document(content=d, type_=language_v1.Document.Type.PLAIN_TEXT)
#                 annotations = client.analyze_sentiment(request={'document': document})
#                 score = annotations.document_sentiment.score
#                 magnitude = annotations.document_sentiment.magnitude
#                 # score of the sentiment ranges between -1.0(negative) and 1.0(positive)
#                 # score in [-1.0, 1.0] is normalized into the range of [0,1]
#                 norm_score = (score+1)/2.
#                 preds.append({
#                     'score': norm_score,
#                     'label': "POSITIVE" if score>=0.0 else "NEGATIVE"
#                 })
#             # end for
#         # end for
#         return preds

#     @classmethod
#     def sentiment_pred_and_conf(cls, data: List[str]):
#         # score of the sentiment ranges between -1.0(negative) and 1.0(positive)
#         # First, score in [-1.0, 1.0] is normalized into the range of [0,1] 
#         preds = cls.batch_predict(data)
#         pr = np.array([x['score'] if x['label']=='POSITIVE' else 1 - x['score'] for x in preds])
#         pp = np.zeros((pr.shape[0], 3))
#         margin_neutral = 1/3.
#         mn = margin_neutral / 2.
#         neg = pr < 0.5 - mn
#         pp[neg, 0] = 1 - pr[neg]
#         pp[neg, 2] = pr[neg]
#         pos = pr > 0.5 + mn
#         pp[pos, 0] = 1 - pr[pos]
#         pp[pos, 2] = pr[pos]
#         neutral_pos = (pr >= 0.5) * (pr < 0.5 + mn)
#         pp[neutral_pos, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_pos] - 0.5)
#         pp[neutral_pos, 2] = 1 - pp[neutral_pos, 1]
#         neutral_neg = (pr < 0.5) * (pr > 0.5 - mn)
#         pp[neutral_neg, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_neg] - 0.5)
#         pp[neutral_neg, 0] = 1 - pp[neutral_neg, 1]
#         preds = np.argmax(pp, axis=1)
#         return preds, pp
    
#     @classmethod
#     def run(cls, testsuite, pred_and_conf_fn, n=Macros.nsamples):
#         testsuite.run(pred_and_conf_fn, n=n, overwrite=True)
#         testsuite.summary(n=100)
#         return
        
    
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(
# #         description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
# #     )
# #     parser.add_argument(
# #         "movie_review_filename",
# #         help="The filename of the movie review you'd like to analyze.",
# #     )
# #     args = parser.parse_args()

# #     analyze(args.movie_review_filename)
