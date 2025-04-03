import json
import numpy as np
import tiktoken

from rank_bm25 import BM25Okapi

DocsCollection = list[dict[str, str]]
SearchResult = dict[str, str | float]

TOP_N = 10

def read_docs() -> DocsCollection:
    with open("enron-search-data.json", "r") as file:
        docs = json.load(file)

    return docs


def build_index(encoder: tiktoken.Encoding, docs: DocsCollection) -> BM25Okapi:
    tokens = map(lambda x: encoder.encode(x["extracted_text"]), docs)
    bm25 = BM25Okapi(tokens)

    return bm25


def search(
    user_query: str,
    enc: tiktoken.Encoding,
    bm25:BM25Okapi,
    docs: DocsCollection,
) -> list[SearchResult]:
    query_enc = enc.encode(user_query)
    doc_scores = bm25.get_scores(query_enc)
    indices = np.argsort(doc_scores)[-TOP_N:][::-1]
    return [
        {
            "docid": docs[idx]["docid"],
            "score": doc_scores[idx],
            "extracted_text": docs[idx]["extracted_text"]
        }
        for idx in indices
    ]


def print_search_result(search_result: SearchResult):
    print(f"Doc ID: {search_result['docid']}")
    print(f"Score: {search_result['score']}")
    print(f"Extracted Text: \n{search_result['extracted_text']}")
    print("="*80)


def main():
    docs = read_docs()
    enc = tiktoken.get_encoding("o200k_base")
    bm25 = build_index(enc, docs)

    while True:
        user_query = input("Your query: ")
        print(f"You entered: {user_query}")
        print(f"Here are the top {TOP_N} results for your query:\n")
        for res in search(user_query, enc, bm25, docs):
            print_search_result(res)

if __name__ == "__main__":
    main()
