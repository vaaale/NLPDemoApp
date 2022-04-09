from backend import load_corpus, load_bi_encoder, load_cross_encoder, embed_corpus, embed_corpus_bm25, bm25_tokenizer, search
from flask import jsonify


passages = load_corpus(wikipedia_filepath='../Data/simplewiki-2020-11-01.jsonl.gz')
bi_encoder = load_bi_encoder()
cross_encoder = load_cross_encoder()

corpus_embeddings = embed_corpus(passages, corpus_embedding_filename="../Data/corpus_embeddings.pk", bi_encoder=bi_encoder)
bm25 = embed_corpus_bm25(passages, bm25_filename="../Data/bm25_embeddings.pk")

print("Backend initialized!")


def score(query, top_k=32):
    response = search(query, passages, corpus_embeddings, bm25, bi_encoder, cross_encoder, top_k=top_k)
    return response

