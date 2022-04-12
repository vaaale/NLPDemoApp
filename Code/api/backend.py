import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
import pickle

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
import umap
import pandas as pd
from sklearn.preprocessing import StandardScaler


if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")


def load_corpus(wikipedia_filepath='../Data/simplewiki-2020-11-01.jsonl.gz'):
    if not os.path.exists(wikipedia_filepath):
        util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

    passages = []
    with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            data = json.loads(line.strip())

            # Add all paragraphs
            # passages.extend(data['paragraphs'])

            #Only add the first paragraph
            passages.append(data['paragraphs'][0])

    print("Passages:", len(passages))
    return passages


def load_bi_encoder(model_name='multi-qa-MiniLM-L6-cos-v1', model_path="../Models/bi_encoder"):
    # We use the Bi-Encoder to encode all passages, so that we can use it with semantic search
    if not os.path.isfile(model_path):
        bi_encoder = SentenceTransformer(model_name)
    else:
        bi_encoder = SentenceTransformer(model_path)
    bi_encoder.eval()
    bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
    return bi_encoder


def load_cross_encoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', model_path="../Models/cross_encoder"):
    # The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    if not os.path.isfile(model_path):
        cross_encoder = CrossEncoder(model_name)
    else:
        cross_encoder = CrossEncoder(model_path)
    cross_encoder.eval()
    return cross_encoder


def embed_corpus(passages, corpus_embedding_filename="../Data/corpus_embeddings.pk", bi_encoder=None):
    if not os.path.isfile(corpus_embedding_filename):
        bi_encoder = load_bi_encoder()
        # As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
        # about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

        # wikipedia_filepath = '../Data/simplewiki-2020-11-01.jsonl.gz'
        #
        # passages = load_corpus(wikipedia_filepath)
        #
        # We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
        corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True, batch_size=512)
        with open(corpus_embedding_filename, "wb") as out:
            pickle.dump(corpus_embeddings, out)
    else:
        with open(corpus_embedding_filename, "rb") as inp:
            corpus_embeddings = pickle.load(inp)

    return corpus_embeddings


def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


def embed_corpus_bm25(passages, bm25_filename="../Data/bm25.pk"):
    # We also compare the results to lexical search (keyword search). Here, we use
    # the BM25 algorithm which is implemented in the rank_bm25 package.
    # We lower case our text and remove stop-words from indexing

    if not os.path.isfile(bm25_filename):
        tokenized_corpus = []
        for passage in tqdm(passages):
            tokenized_corpus.append(bm25_tokenizer(passage))

        bm25 = BM25Okapi(tokenized_corpus)
        with open(bm25_filename, "wb") as out:
            pickle.dump(bm25, out)
    else:
        with open(bm25_filename, "rb") as inp:
            bm25 = pickle.load(inp)

    return bm25


# This function will search all wikipedia articles for passages that
# answer the query
def search(query, passages, corpus_embeddings, bm25, bi_encoder, cross_encoder, top_k=32):
    df = pd.DataFrame()
    print("Input question:", query)

    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -10)[-10:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    print("Top-3 lexical search (BM25) hits")
    for hit in bm25_hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    bm25_answers = [query]
    bm25_scores = [1.]
    for hit in bm25_hits:
        h = hit["score"]
        score = f"{h:.3f}"
        answer = passages[hit['corpus_id']].replace("\n", " ")
        bm25_answers.append(answer)
        bm25_scores.append(score)
    bm25_answers += [""] * (top_k - len(bm25_answers)+1)
    bm25_scores += [0] * (top_k - len(bm25_scores)+1)
    df["bm25"] = bm25_answers
    df["bm25_scores"] = bm25_scores


    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    bi_encoder_answers = [query]
    bi_encoder_scores = [1.]
    emb = [question_embedding.cpu().numpy()]
    for hit in hits:
        h = hit["score"]
        # score = f"{h:.3f}"
        score = h
        answer = passages[hit['corpus_id']].replace("\n", " ")
        bi_encoder_answers.append(answer)
        bi_encoder_scores.append(score)
        emb.append(corpus_embeddings[hit['corpus_id']].cpu().numpy())
    # emb = reduce(emb)
    df["bi_encoder"] = bi_encoder_answers
    df["bi_encoder_scores"] = bi_encoder_scores
    # df["bi_encoder_emb_X"] = emb[:, 0]
    # df["bi_encoder_emb_Y"] = emb[:, 1]


    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))

    cross_encoder_answers = [query]
    cross_encoder_scores = [1.]
    for hit in hits:
        h = hit["cross-score"]
        score = f"{h:.3f}"
        answer = passages[hit['corpus_id']].replace("\n", " ")
        cross_encoder_answers.append(answer)
        cross_encoder_scores.append(score)
    df["cross_encoder"] = cross_encoder_answers
    df["cross_encoder_scores"] = cross_encoder_scores

    return df, emb


def reduce(embeddings):
    print("Reducing embeddings....")
    scaler = StandardScaler()
    reducer = umap.UMAP(n_components=2)
    reduced = reducer.fit_transform(scaler.fit_transform(embeddings))
    # m = np.array(embeddings)
    # d = m.T @ m
    # norm = (m * m).sum(0, keepdims=True) ** .5
    # sim = d / norm / norm.T
    # dist = 1 - d / norm / norm.T
    print("Done!")
    return reduced

