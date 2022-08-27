"""Server file
Attributes:
    app (fastapi.applications.FastAPI): Fast API app
"""

import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

load_dotenv()
from core.reranker import ConceptMatchRanker
from core.custom_reranker import CustomRanker

rerankers = {
    "concept-match-ranker": ConceptMatchRanker(),
    "custom-ranker": CustomRanker(),
}


class RerankingRequest(BaseModel):
    query: str
    model: str
    docs: List[str]

class ScoringRequest(BaseModel):
    query: str
    model: str
    doc: str


app = FastAPI()


@app.post("/rerank")
async def rerank(req: RerankingRequest):
    """Sort documents as per their similarity with a query"""
    if not req.docs:
        raise HTTPException(status_code=400, detail="No documents provided")

    reranker = rerankers.get(req.model)
    if not reranker:
        raise HTTPException(status_code=400, detail="Invalid model name")

    return {"ranks": reranker.rank(req.query, req.docs).tolist()}


@app.post("/score")
async def score(req: ScoringRequest):
    """Sort documents as per their similarity with a query"""
    if not req.doc:
        raise HTTPException(status_code=400, detail="No document!")

    reranker = rerankers.get(req.model)
    if not reranker:
        raise HTTPException(status_code=400, detail="Invalid model name")

    return {"score": reranker.score(req.query, req.doc)}


if __name__ == "__main__":
    port = int(os.environ["PORT"])
    uvicorn.run(app, host="0.0.0.0", port=port)
