from api.backend import load_corpus, load_bi_encoder, load_cross_encoder, embed_corpus, embed_corpus_bm25, search
from transformers import pipeline
from sentence_transformers import util

passages = load_corpus(wikipedia_filepath='../Data/simplewiki-2020-11-01.jsonl.gz')
bi_encoder = load_bi_encoder()
cross_encoder = load_cross_encoder()

corpus_embeddings = embed_corpus(passages, corpus_embedding_filename="../Data/corpus_embeddings.pk", bi_encoder=bi_encoder)
bm25 = embed_corpus_bm25(passages, bm25_filename="../Data/bm25_embeddings.pk")


qa_model = pipeline("question-answering")
print("Backend initialized!")


def score(query, top_k=32):
    response, embeddings = search(query, passages, corpus_embeddings, bm25, bi_encoder, cross_encoder, top_k=top_k)
    return response, embeddings


def exact_answer(query, top_k=32):
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

    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    bi_encoder_answers = []
    for hit in hits:
        answer = passages[hit['corpus_id']].replace("\n", " ")
        bi_encoder_answers.append(answer)

    context = " ".join(bi_encoder_answers)
    response = qa_model(question=query, context=context)
    print(response)
    return response, context



