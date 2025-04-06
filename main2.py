import json
import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

DocsCollection = list[dict[str, str]]
SearchResult = dict[str, str | float]

TOP_N = 10

def read_docs() -> DocsCollection:
    with open("enron-search-data.json", "r") as file:
        docs = json.load(file)
    return docs

def build_indices(docs: DocsCollection) -> tuple:
    enc = tiktoken.get_encoding("o200k_base")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [doc["extracted_text"] for doc in docs]
    tokens = list(map(lambda x: enc.encode(x), texts))
    bm25 = BM25Okapi(tokens)
    
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return (bm25, index, model, enc)

def hybrid_search(query: str, bm25: BM25Okapi, index: faiss.Index, 
                 model: SentenceTransformer, enc: tiktoken.Encoding,
                 docs: DocsCollection, alpha: float = 0.5) -> list[SearchResult]:
    query_enc = enc.encode(query)
    bm25_scores = np.array(bm25.get_scores(query_enc))
    bm25_scores = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores
    
    query_vector = model.encode([query])[0].astype('float32').reshape(1, -1)
    distances, semantic_indices = index.search(query_vector, len(docs))
    semantic_scores = 1.0 / (1.0 + distances[0])
    
    combined_scores = {}
    for i, score in enumerate(bm25_scores):
        combined_scores[i] = alpha * score
        
    for i, idx in enumerate(semantic_indices[0]):
        if idx in combined_scores:
            combined_scores[idx] += (1-alpha) * semantic_scores[i]
        else:
            combined_scores[idx] = (1-alpha) * semantic_scores[i]
            
    results = [(idx, score) for idx, score in combined_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return [
        {
            "docid": docs[idx]["docid"],
            "score": score,
            "extracted_text": docs[idx]["extracted_text"]
        }
        for idx, score in results[:TOP_N]
    ]

def main():
    docs = read_docs()
    bm25, index, model, enc = build_indices(docs)
    
    with open('query2.txt', 'r') as file:
        user_query = file.read().strip()
        print(f"You entered: {user_query}")
    
    for res in hybrid_search(user_query, bm25, index, model, enc, docs):
        print(f"Doc ID: {res['docid']}")
        print(f"Score: {res['score']}")
        print("="*80)

if __name__ == "__main__":
    main()