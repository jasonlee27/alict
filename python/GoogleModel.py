"""Demonstrates how to make a simple call to the Natural Language API."""
# Score of the sentiment ranges between -1.0 (negative) and 1.0 (positive)
# and corresponds to the overall emotional leaning of the text.
# 
# Magnitude indicates the overall strength of emotion (both positive and negative)
# within the given text, between 0.0 and +inf. Unlike score, magnitude is not normalized;
# each expression of emotion within the text (both positive and negative)
# contributes to the text's magnitude (so longer text blocks may have greater magnitudes).

import argparse

from google.cloud import language_v1


class GoogleModel:

    def load_model_client(cls):
        return language_v1.LanguageServiceClient()
        
    @classmethod
    def get_batch(cls, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
        # end for


    @classmethod
    def batch_predict(cls, data: List[str], batch_size: int = 32):
        preds = list()
        for d in cls.get_batch(data, batch_size):
            preds.extend(cls.model(d))
        # end for
        return preds

def print_result(annotations, sent_idx, sent):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print(
            "Sentence {} has a sentiment score of {}".format(index, sentence_sentiment)
        )

    print(
        "Overall Sentiment: score of {} with magnitude of {}".format(score, magnitude)
    )
    return 0




def analyze(movie_review_filename):
    """Run a sentiment analysis request on text within a passed filename."""
    client = language_v1.LanguageServiceClient()

    with open(movie_review_filename, "r") as review_file:
        # Instantiates a plain text document.
        sents = [l.strip() for l in review_file.readlines()]

    for s_i, s in enumerate(sents):
        document = language_v1.Document(content=s, type_=language_v1.Document.Type.PLAIN_TEXT)
        annotations = client.analyze_sentiment(request={'document': document})
        
        # Print the results
        print_result(annotations, s_i, s)
    # end for





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "movie_review_filename",
        help="The filename of the movie review you'd like to analyze.",
    )
    args = parser.parse_args()

    analyze(args.movie_review_filename)
