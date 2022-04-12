import pandas as pd
import nlp
import pickle
import faiss
import numpy as np
import os

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from lfqa_utils import query_qa_dense_index, qa_s2s_generate


def load_datasets():
    with open("../Data/wiki_snippets.pk", "rb") as inp:
        wiki40b_snippets = pickle.load(inp)
    eli5 = nlp.load_dataset('eli5')
    return wiki40b_snippets, eli5


def build_faiss_index(wiki40b_snippets):
    index_file = "../Data/wiki40b_index_flat.ind"
    if not os.path.isfile(index_file):
        faiss_res = faiss.StandardGpuResources()
        wiki40b_passage_reps = np.memmap(
            'wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat',
            dtype='float32', mode='r',
            shape=(wiki40b_snippets.num_rows, 128)
        )

        wiki40b_index_flat = faiss.IndexFlatIP(128)
        faiss.write_index(wiki40b_index_flat, index_file)
        wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)
        wiki40b_gpu_index.add(wiki40b_passage_reps)
    else:
        wiki40b_index_flat = faiss.read_index("../Data/wiki40b_index_flat.ind")
        faiss_res = faiss.StandardGpuResources()
        wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)
    return wiki40b_gpu_index


def load_qar_model():
    qar_tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased')
    qar_model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to('cuda:0')
    _ = qar_model.eval()
    return qar_tokenizer, qar_model


def load_qa_s2s_model():
    qa_s2s_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
    qa_s2s_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')
    _ = qa_s2s_model.eval()
    return qa_s2s_tokenizer, qa_s2s_model


def test(wiki40b_snippets, eli5, wiki40b_gpu_index, qar_tokenizer, qar_model, qa_s2s_tokenizer, qa_s2s_model):
    questions = []
    answers = []

    for i in [12345] + [j for j in range(4)]:
        # create support document with the dense index
        question = eli5['test_eli5'][i]['title']
        doc, res_list = query_qa_dense_index(
            question, qar_model, qar_tokenizer,
            wiki40b_snippets, wiki40b_gpu_index, device='cuda:0'
        )
        # concatenate question and support document into BART input
        question_doc = "question: {} context: {}".format(question, doc)
        # generate an answer with beam search
        answer = qa_s2s_generate(
            question_doc, qa_s2s_model, qa_s2s_tokenizer,
            num_answers=1,
            num_beams=8,
            min_len=64,
            max_len=256,
            max_input_length=1024,
            device="cuda:0"
        )[0]
        questions += [question]
        answers += [answer]

    df = pd.DataFrame({
        'Question': questions,
        'Answer': answers,
    })
    df.style.set_properties(**{'text-align': 'left'})
    return df


def test2(question, wiki40b_snippets, wiki40b_gpu_index, qar_tokenizer, qar_model, qa_s2s_tokenizer, qa_s2s_model):
    doc, res_list = query_qa_dense_index(
        question, qar_model, qar_tokenizer,
        wiki40b_snippets, wiki40b_gpu_index, device='cuda:0'
    )
    # concatenate question and support document into BART input
    question_doc = "question: {} context: {}".format(question, doc)
    # generate an answer with beam search
    answer = qa_s2s_generate(
        question_doc, qa_s2s_model, qa_s2s_tokenizer,
        num_answers=1,
        num_beams=8,
        min_len=64,
        max_len=256,
        max_input_length=1024,
        device="cuda:0"
    )[0]
    return question, answer


def main():
    print("Loading datasets....")
    wiki, eli5 = load_datasets()
    print("Loading Faiss index....")
    wiki40b_gpu_index = build_faiss_index(wiki)
    print("Loading QAR model....")
    qar_tokenizer, qar_model = load_qar_model()
    print("Loading QA S2S model....")
    qa_s2s_tokenizer, qa_s2s_model = load_qa_s2s_model()
    print("Running test.....")
    df = test(wiki, eli5, wiki40b_gpu_index, qar_tokenizer, qar_model, qa_s2s_tokenizer, qa_s2s_model)
    print(df)
    print("\nRunning single question....\n")
    question = "Why does water heated to room temperature feel colder than the air around it?"
    question, answer = test2(question, wiki, wiki40b_gpu_index, qar_tokenizer, qar_model, qa_s2s_tokenizer, qa_s2s_model)
    print(f"\tQuestion: {question}\n\tAnswer: {answer}")


if __name__ == '__main__':
    main()
