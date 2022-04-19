import nlp
import pickle
import faiss
import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from lfqa_utils import query_qa_dense_index, qa_s2s_generate


class AbstractiveAPI:
    def __init__(self):
        with open("../Data/wiki_snippets.pk", "rb") as inp:
            self.wiki40b_snippets = pickle.load(inp)
        self.eli5 = nlp.load_dataset('eli5')

        faiss_res = faiss.StandardGpuResources()
        wiki40b_passage_reps = np.memmap(
            '../Data/wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat',
            dtype='float32', mode='r',
            shape=(self.wiki40b_snippets.num_rows, 128)
        )

        wiki40b_index_flat = faiss.IndexFlatIP(128)
        self.wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)
        self.wiki40b_gpu_index.add(wiki40b_passage_reps)

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




