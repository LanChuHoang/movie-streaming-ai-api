from . import vectorizer
from .model import model


def sentiment_analysis_pipeline(reviews):
    cleaned_comments = [vectorizer.process_review(review) for review in reviews]
    vecs = vectorizer.transform(cleaned_comments)
    sentiments = model.predict(vecs)
    return sentiments
