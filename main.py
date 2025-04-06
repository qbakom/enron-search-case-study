import json
import numpy as np
import tiktoken

from rank_bm25 import BM25Okapi

# import faiss
# from sentence_transformers import SentenceTransformer


DocsCollection = list[dict[str, str]]
SearchResult = dict[str, str | float]

TOP_N = 10

def read_docs() -> DocsCollection:
    with open("enron-search-data.json", "r") as file:
        docs = json.load(file)

    return docs


# def build_index(encoder: tiktoken.Encoding, docs: DocsCollection) -> BM25Okapi:
#     tokens = map(lambda x: encoder.encode(x["extracted_text"]), docs)
#     bm25 = BM25Okapi(tokens)

#     return bm25

def build_index(encoder: tiktoken.Encoding, docs: DocsCollection) -> tuple:
    # Initialize sentence transformer model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract text from documents
    texts = [doc["extracted_text"] for doc in docs]
    
    # Generate document embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Convert to float32 as required by FAISS
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return (index, model)


def search(
    user_query: str,
    # enc: tiktoken.Encoding,
    # bm25:BM25Okapi,
    model: SentenceTransformer,
    index: faiss.Index,
    docs: DocsCollection,
) -> list[SearchResult]:
    # query_enc = enc.encode(user_query)
    # doc_scores = bm25.get_scores(query_enc)
    # indices = np.argsort(doc_scores)[-TOP_N:][::-1]
    # return [
    #     {
    #         "docid": docs[idx]["docid"],
    #         "score": doc_scores[idx],
    #         "extracted_text": docs[idx]["extracted_text"]
    #     }
    #     for idx in indices
    # ]
    # Generate embedding for the query
    query_vector = model.encode([user_query])[0].astype('float32').reshape(1, -1)
    
    # Search the index
    distances, indices = index.search(query_vector, TOP_N)
    
    return [
        {
            "docid": docs[idx]["docid"],
            "score": float(1.0 / (1.0 + distances[0][i])),  # Convert distance to similarity score
            "extracted_text": docs[idx]["extracted_text"]
        }
        for i, idx in enumerate(indices[0])
    ]


def print_search_result(search_result: SearchResult):
    print(f"Doc ID: {search_result['docid']}")
    print(f"Score: {search_result['score']}")
    # print(f"Extracted Text: \n{search_result['extracted_text']}")
    print("="*80)


def main():
    bm25 = build_index(enc, docs)
    docs = read_docs()
    enc = tiktoken.get_encoding("o200k_base")
    index, model = build_index(enc, docs)

    #while True:
    with open('query2.txt', 'r') as file:
        user_query = file.read().strip()
        print(f"You entered: {user_query}")
        
    for res in search(user_query, model, index, docs):
        print_search_result(res)
        
        
        
    for res in search(user_query, enc, bm25, docs):
        print_search_result(res)
    # user_query = input("Your query: ")
    # print(f"You entered: {user_query}")
    # print(f"Here are the top {TOP_N} results for your query:\n")
    # for res in search(user_query, enc, bm25, docs):
    #     print_search_result(res)
    
    # result = ""
    # for res in search(user_query, enc, bm25, docs):
    #     result += res['docid'] + "\n"
    
    # with open('result.txt', 'w') as file:
    #     file.write(result)
    # file.close()
    
    # search_result = search(user_query, enc, bm25, docs)
    # with open("query2.txt", "w") as file:
    #     file.write(search_result[0]["extracted_text"])

if __name__ == "__main__":
    main()
