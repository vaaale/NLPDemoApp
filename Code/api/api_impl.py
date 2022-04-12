from api.backend import load_corpus, load_bi_encoder, load_cross_encoder, embed_corpus, embed_corpus_bm25, search, reduce
from transformers import pipeline
from sentence_transformers import util

import nlp
import pickle
import faiss
import numpy as np
import os

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from lfqa_utils import query_qa_dense_index, qa_s2s_generate


class AbstractiveAPI:
    def __init__(self):
        with open("../Data/wiki_snippets.pk", "rb") as inp:
            self.wiki40b_snippets = pickle.load(inp)
        self.eli5 = nlp.load_dataset('eli5')

        index_file = "../Data/wiki40b_index_flat.ind"
        if not os.path.isfile(index_file):
            faiss_res = faiss.StandardGpuResources()
            wiki40b_passage_reps = np.memmap(
                '../Data/wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat',
                dtype='float32', mode='r',
                shape=(self.wiki40b_snippets.num_rows, 128)
            )

            wiki40b_index_flat = faiss.IndexFlatIP(128)
            self.wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)
            self.wiki40b_gpu_index.add(wiki40b_passage_reps)
            faiss.write_index(wiki40b_index_flat, index_file)
        else:
            wiki40b_index_flat = faiss.read_index(index_file)
            faiss_res = faiss.StandardGpuResources()
            self.wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)

        self.qar_tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased')
        self.qar_model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to('cuda:0')
        _ = self.qar_model.eval()

        self.qa_s2s_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
        self.qa_s2s_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')
        _ = self.qa_s2s_model.eval()

    def answer(self, question):
        doc, res_list = query_qa_dense_index(
            question, self.qar_model, self.qar_tokenizer,
            self.wiki40b_snippets, self.wiki40b_gpu_index, device='cuda:0'
        )
        # concatenate question and support document into BART input
        question_doc = "question: {} context: {}".format(question, doc)
        # generate an answer with beam search
        answer = qa_s2s_generate(
            question_doc, self.qa_s2s_model, self.qa_s2s_tokenizer,
            num_answers=1,
            num_beams=8,
            min_len=64,
            max_len=256,
            max_input_length=1024,
            device="cuda:0"
        )[0]
        print("Sample:")
        print(question_doc)
        return answer, doc


class ExtractiveAPI:
    def __init__(self):
        self.passages = load_corpus(wikipedia_filepath='../Data/simplewiki-2020-11-01.jsonl.gz')
        self.bi_encoder = load_bi_encoder()
        self.cross_encoder = load_cross_encoder()

        self.corpus_embeddings = embed_corpus(self.passages, corpus_embedding_filename="../Data/corpus_embeddings.pk", bi_encoder=self.bi_encoder)
        self.bm25 = embed_corpus_bm25(self.passages, bm25_filename="../Data/bm25_embeddings.pk")
        self.qa_model = pipeline("question-answering")
        print("Backend initialized!")

    def score(self, query, top_k=32):
        response, embeddings = search(query, self.passages, self.corpus_embeddings, self.bm25, self.bi_encoder, self.cross_encoder, top_k=top_k)
        return response, embeddings

    def exact_answer(self, query, top_k=32):
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=top_k)
        hits = hits[0]  # Get the hits for the first query

        ##### Re-Ranking #####
        # Now, score all retrieved passages with the cross_encoder
        cross_inp = [[query, self.passages[hit['corpus_id']]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_inp)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        bi_encoder_answers = []
        for hit in hits:
            answer = self.passages[hit['corpus_id']].replace("\n", " ")
            bi_encoder_answers.append(answer)

        context = " ".join(bi_encoder_answers)
        response = self.qa_model(question=query, context=context)
        print(response)
        return response, context

    def reduce(self, embeddings):
        return reduce(embeddings)



