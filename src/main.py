from typing import List

from fastapi import FastAPI

from .model.pipeline import sentiment_analysis_pipeline

app = FastAPI()


@app.post("/sentiment")
async def post_items(reviews: List[str]):
    sentiments = sentiment_analysis_pipeline(reviews)
    return {"result": sentiments.tolist()}
