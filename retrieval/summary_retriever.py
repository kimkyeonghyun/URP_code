#!/usr/bin/env python
# coding: utf-8

# --- 라이브러리 임포트 ---
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from ast import literal_eval
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
import re
import ast

# --- 데이터 로드 및 전처리 ---
def load_data():
    """
    데이터 로드 및 필터링 함수.
    - 민원 데이터와 법령 데이터를 불러오고 필요한 데이터를 필터링.
    """
    law_question = pd.read_csv('./law_data/law_hang_summary.csv', encoding='utf-8-sig', lineterminator='\n')
    with open('./minwon_law_list.txt', 'r', encoding='utf-8-sig') as f:
        minwon_law_list = [line.strip() for line in f.readlines()]
    law_question = law_question[law_question['law'].isin(minwon_law_list)].reset_index()
    law_question.to_csv('./law_data/law_filter_summer.csv', index=False)

    # 법령 데이터 로드 및 분리
    passage = pd.read_csv('./law_data/law_filter_raw.csv', encoding='utf-8-sig', lineterminator='\n')
    passage[['law_name', 'law_clause']] = passage['passage'].apply(lambda x: pd.Series(split_passage(x)))

    # 민원 데이터 로드 및 정리
    question = pd.read_csv('../민원22~24년도통합_test_1010_law_yoji.csv', encoding='utf-8-sig', lineterminator='\n')
    question['민원요지'] = question['민원요지'].apply(lambda x: x.split('민원요지:')[-1].strip() if isinstance(x, str) else x)
    question = question.dropna(subset=['민원요지']).reset_index()
    question['추출된 법(re)'] = question['추출된 법(re)'].apply(ast.literal_eval).tolist()

    return law_question, passage, question

def split_passage(text):
    """
    Passage 데이터를 법 이름과 조항으로 분리.
    """
    parts = text.split(' : ', 1)
    if len(parts) < 2:
        return '', ''
    law_parts = parts[0].split(' ; ')
    if len(law_parts) == 3:
        return law_parts[0].strip(), ' '.join(law_parts[1:]).strip()
    elif len(law_parts) == 2:
        return law_parts[0].strip(), law_parts[1].strip()
    return '', ''

# --- 유틸리티 함수 ---
def average_pool(last_hidden_states, attention_mask):
    """
    Transformer 출력에서 평균 풀링을 수행하여 임베딩을 생성.
    """
    masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def process_and_combine_results(results):
    """
    검색 결과를 전처리하고 결합.
    """
    combined_results = []
    for res in results:
        split_res = res.split(' : ')[0].split(' ; ')
        if '부칙' in split_res:
            combined_results.append(f"{split_res[0]} ; {split_res[1]} ; {split_res[2]}")
        else:
            combined_results.append(f"{split_res[0]} ; {split_res[1]}")
    return combined_results

def remove_parentheses(text):
    """
    텍스트에서 괄호를 제거하고 나머지 내용을 반환.
    """
    return re.sub(r'\([^)]*\)', '', text).strip().split(' ; ')

def split_law_and_article(text):
    # '제숫자조' 뒤에 의2와 같은 추가 내용을 포함하도록 정규식 수정
    match = re.match(r'(.*?)(제\d+조[\w]*)', text)
    if match:
        law_name = match.group(1).replace(' ', '').strip()  # 법 이름 부분
        article = match.group(2).strip()  # 조항 부분
        return [law_name, article]
    else:
        return [text.strip(), '']

def exp_normalize(x):
    """
    값을 정규화하여 확률 분포 형태로 변환.
    """
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

# --- 모델 및 토크나이저 ---
def load_model_and_tokenizer(model_name, device):
    """
    지정된 모델과 토크나이저를 로드.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model

def preprocess_passages(data, tokenizer, field, max_length=512):
    """
    Passage 데이터를 토큰화하여 모델 입력 형태로 변환.
    """
    passages = {'passages': [], 'passages_attention_mask': []}
    for text in tqdm(data[field], desc=f'Preprocessing {field}'):
        tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        passages['passages'].append(tokenized['input_ids'].squeeze(0))
        passages['passages_attention_mask'].append(tokenized['attention_mask'].squeeze(0))
    return passages

def compute_passage_embeddings(passages, model, device):
    """
    Passage 데이터로부터 DPR 임베딩을 생성.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(passages['passages'])), desc='Computing embeddings'):
            passage = passages['passages'][i].unsqueeze(0).to(device)
            attention_mask = passages['passages_attention_mask'][i].unsqueeze(0).to(device)
            output = model(input_ids=passage, attention_mask=attention_mask)
            embeddings.append(average_pool(output.last_hidden_state, attention_mask))
    return torch.cat(embeddings, dim=0)

def check_match(dpr_cleaned, split_laws_cleaned):
    """
    검색된 결과 중 법 이름과 조항 번호가 모두 일치하는지 확인.
    """
    return int(any(law[0] == item[0] and law[1] == item[1] for law in split_laws_cleaned for item in dpr_cleaned))

def check_all_match(dpr_cleaned, split_laws_cleaned):
    """
    검색된 결과가 모두 일치하는지 확인.
    """
    for law in split_laws_cleaned:
        if not any(law[0] == item[0] and law[1] == item[1] in item for item in dpr_cleaned):
            return 0
    return 1
# --- 검색 및 재정렬 ---
def get_all_results(question, q_tokenizer, q_model, p_embs, law_question, passage, corpus, kiwi_tokenizer, device):
    """
    DPR 및 BM25를 사용하여 질문에 대한 관련 Passage 검색.
    """
    question_token = q_tokenizer(question, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    q_model.eval()
    with torch.no_grad():
        question_output = q_model(input_ids=question_token['input_ids'].to(device),
                                  attention_mask=question_token['attention_mask'].to(device))
        question_embedding = average_pool(question_output.last_hidden_state, question_token['attention_mask'].to(device))

    # DPR 결과 계산
    similarity_scores = F.cosine_similarity(question_embedding.expand(p_embs.size(0), -1), p_embs, dim=1)
    rank = torch.argsort(similarity_scores, descending=True)
    dpr_results = []
    for idx in rank[:50]:
        result = passage[(passage['law_name'] == law_question.iloc[idx.item()]['law']) & 
                         (passage['law_clause'] == law_question.iloc[idx.item()]['article'])]
        if not result.empty:
            dpr_results.append(result['passage'].iloc[0])
            if len(dpr_results) >= 50:
                break

    # BM25 결과 계산
    bm25_results = []
    tokenized_question = kiwi_tokenizer(question)
    bm25_tokenized_results = retriever.get_top_n(tokenized_question, list(range(len(corpus))), len(corpus))
    for idx in bm25_tokenized_results[:50]:
        result = passage[(passage['law_name'] == law_question.iloc[idx]['law']) & 
                         (passage['law_clause'] == law_question.iloc[idx]['article'])]
        if not result.empty:
            bm25_results.append(result['passage'].iloc[0])
            if len(bm25_results) >= 50:
                break
    return dpr_results, bm25_results

def rerank_with_reranker(dpr_results, bm25_results, question, reranker_tokenizer, reranker_model):
    """
    DPR 및 BM25 결과를 Reranker로 재정렬.
    """
    combined_results = list(set(dpr_results + bm25_results))
    pairs = [[question, passage] for passage in combined_results]
    tokenized_inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        logits = reranker_model(**tokenized_inputs).logits.view(-1).cpu().numpy()

    scores = exp_normalize(logits)
    reranked = sorted(zip(combined_results, scores), key=lambda x: x[1], reverse=True)
    return [res[0] for res in reranked]

# --- 메인 실행 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
law_question, passage, question = load_data()
q_tokenizer, q_model = load_model_and_tokenizer('nlpai-lab/KoE5', device)
reranker_tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')
reranker_model = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker').to(device)

passages = preprocess_passages(law_question, q_tokenizer, 'summary')
p_embs = compute_passage_embeddings(passages, q_model, device)

# BM25 설정
kiwi = Kiwi()
stopwords = Stopwords()
def kiwi_tokenizer(doc):
    return [token.form for token in kiwi.tokenize(doc, normalize_coda=True, stopwords=stopwords)]

corpus = [kiwi_tokenizer(doc) for doc in law_question['summary']]
retriever = BM25Okapi(corpus)

# --- 점수 평가 ---
top_k_values = [1, 5, 10]
all_check_match_dpr_results = {k: 0 for k in top_k_values}
all_check_match_bm25_results = {k: 0 for k in top_k_values}
all_check_match_reranked_results = {k: 0 for k in top_k_values}
all_check_all_match_dpr_results = 0
all_check_all_match_bm25_results = 0
all_check_all_match_reranked_results = 0

question['dpr_result'] = None
question['bm25_result'] = None
question['reranking_result'] = None


# 질의마다 처리
for idx in tqdm(range(len(question)), desc='Retrieving questions'):
    dpr_results, bm25_results = get_all_results(
        question['민원요지'][idx], q_tokenizer, q_model, p_embs, law_question, passage, corpus, kiwi_tokenizer, device)
    reranked_results = rerank_with_reranker(dpr_results[:50], bm25_results[:50], question['민원요지'][idx], reranker_tokenizer, reranker_model)

    # 결과 전처리
    process_dpr_results = process_and_combine_results(dpr_results[:10])
    process_bm25_results = process_and_combine_results(bm25_results[:10])
    process_reranked_results = process_and_combine_results(reranked_results[:10])

    dpr_cleaned = [remove_parentheses(item) for item in process_dpr_results]
    bm25_cleaned = [remove_parentheses(item) for item in process_bm25_results]
    reranked_cleaned = [remove_parentheses(item) for item in process_reranked_results]

    split_laws = [split_law_and_article(item) for item in question['추출된 법(re)'][idx]]

    # 저장된 결과 업데이트 (top-10 기준)
    question.at[idx, 'dpr_result'] = dpr_cleaned
    question.at[idx, 'bm25_result'] = bm25_cleaned
    question.at[idx, 'reranking_result'] = reranked_cleaned

    # top-k 평가
    for k in top_k_values:
        all_check_match_dpr_results[k] += check_match(dpr_cleaned[:k], split_laws)
        all_check_match_bm25_results[k] += check_match(bm25_cleaned[:k], split_laws)
        all_check_match_reranked_results[k] += check_match(reranked_cleaned[:k], split_laws)

    # 전체 일치 평가 (top-10)
    all_check_all_match_dpr_results += check_all_match(dpr_cleaned[:10], split_laws)
    all_check_all_match_bm25_results += check_all_match(bm25_cleaned[:10], split_laws)
    all_check_all_match_reranked_results += check_all_match(reranked_cleaned[:10], split_laws)

# --- 최종 평가 결과 출력 ---
print("\n--- Top-k Accuracy ---")
for k in top_k_values:
    print(f"Top-{k} DPR Accuracy: {all_check_match_dpr_results[k] / len(question):.3f}")
    print(f"Top-{k} BM25 Accuracy: {all_check_match_bm25_results[k] / len(question):.3f}")
    print(f"Top-{k} Reranked Accuracy: {all_check_match_reranked_results[k] / len(question):.3f}")

print("\n--- All Match Accuracy (Top-10) ---")
print(f"DPR All Match Accuracy: {all_check_all_match_dpr_results / len(question):.3f}")
print(f"BM25 All Match Accuracy: {all_check_all_match_bm25_results / len(question):.3f}")
print(f"Reranked All Match Accuracy: {all_check_all_match_reranked_results / len(question):.3f}")

# --- 결과 저장 ---
question.to_csv('./results/summary_retrieval_results.csv', encoding='utf-8-sig', index=False)