# --- 라이브러리 임포트 ---
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch import Tensor
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
import re
import ast

# --- Passage 전처리 ---
def split_passage(text):
    """
    Passage 데이터를 법 이름과 조항으로 분리합니다.
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


# --- 데이터 전처리 ---
def process_and_combine_results(results):
    """
    검색 결과를 전처리하고 결합합니다.
    """
    combined_results = []
    for res in results:
        split_res = res.split(' : ')[0].split(' ; ')
        if '부칙' in split_res:
            combined_results.append(f"{split_res[0]} ; {split_res[1]} ; {split_res[2]}")
        else:
            combined_results.append(f"{split_res[0]} ; {split_res[1]}")
    return combined_results


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    평균 풀링을 사용해 Transformer의 hidden state에서 임베딩을 생성합니다.
    """
    masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def preprocess_passages(passage_data: pd.DataFrame, tokenizer, max_length: int = 512):
    """
    Passage 데이터를 모델 입력 형태로 변환합니다.
    """
    passages = {'passages': [], 'passages_attention_mask': [], 'passage_ids': passage_data.index.tolist()}
    for passage_text in tqdm(passage_data['passage'], desc='Preprocessing passages'):
        tokenized = tokenizer(passage_text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        passages['passages'].append(tokenized['input_ids'].squeeze(0))
        passages['passages_attention_mask'].append(tokenized['attention_mask'].squeeze(0))
    return passages

def exp_normalize(x):
    """
    입력 배열의 확률값을 정규화.
    """
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def compute_passage_embeddings(passages, model, device):
    """
    Passage 데이터를 DPR 임베딩으로 변환합니다.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for idx in tqdm(range(len(passages['passages'])), desc='Computing embeddings'):
            input_ids = passages['passages'][idx].unsqueeze(0).to(device)
            attention_mask = passages['passages_attention_mask'][idx].unsqueeze(0).to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(average_pool(output.last_hidden_state, attention_mask))
    return torch.cat(embeddings, dim=0)


# --- 검색 ---
def get_all_results(question, q_tokenizer, q_model, p_embs, passage_data, retriever, corpus, kiwi_tokenizer, device):
    """
    DPR 및 BM25를 사용해 질문에 대한 검색 결과를 반환합니다.
    """
    question_token = q_tokenizer(question, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        question_output = q_model(input_ids=question_token['input_ids'], attention_mask=question_token['attention_mask'])
        question_embedding = average_pool(question_output.last_hidden_state, question_token['attention_mask'])

    # DPR 검색
    similarity_scores = F.cosine_similarity(question_embedding.expand(p_embs.size(0), -1), p_embs, dim=1)
    dpr_results = [passage_data[idx] for idx in torch.argsort(similarity_scores, descending=True)[:50]]

    # BM25 검색
    bm25_results = [passage_data[idx] for idx in retriever.get_top_n(kiwi_tokenizer(question), list(range(len(corpus))), len(corpus))[:50]]

    return dpr_results, bm25_results


# --- Reranking ---
def rerank_with_reranker(dpr_results, bm25_results, question, reranker_tokenizer, reranker_model):
    """
    DPR 및 BM25 검색 결과를 Reranker 모델을 사용하여 재정렬합니다.
    """
    combined_results = list(set(dpr_results + bm25_results))
    inputs = reranker_tokenizer([[question, doc] for doc in combined_results], padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)

    with torch.no_grad():
        logits = reranker_model(**inputs).logits.view(-1).cpu().numpy()

    reranker_scores = exp_normalize(logits)
    sorted_results = sorted(zip(combined_results, reranker_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_results]


# --- 평가 ---
def remove_parentheses(text):
    """
    괄호를 제거한 후 텍스트를 반환합니다.
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


# --- 메인 실행 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
passage = pd.read_csv('./law_data/law_filter_raw.csv', encoding='utf-8-sig', lineterminator='\n')
passage[['law_name', 'law_clause']] = passage['passage'].apply(lambda x: pd.Series(split_passage(x)))

q_tokenizer, q_model = AutoTokenizer.from_pretrained('nlpai-lab/KoE5'), AutoModel.from_pretrained('nlpai-lab/KoE5').to(device)
reranker_tokenizer = AutoTokenizer.from_pretrained('Dongjin-kr/ko-reranker')
reranker_model = AutoModelForSequenceClassification.from_pretrained('Dongjin-kr/ko-reranker').to(device).eval()

passages = preprocess_passages(passage, q_tokenizer)
p_embs = compute_passage_embeddings(passages, q_model, device)

kiwi = Kiwi()
stopwords = Stopwords()
def kiwi_tokenizer(doc):
    tokens = kiwi.tokenize(doc, normalize_coda=True, stopwords=stopwords)
    return [token.form for token in tokens]

corpus = [kiwi_tokenizer(doc) for doc in tqdm(passage['passage'].tolist(), desc="Tokenizing passages with Kiwi")]
retriever = BM25Okapi(corpus)

questions = pd.read_csv('./민원22~24년도통합_test_1010_law_yoji.csv', encoding='utf-8-sig')
questions['추출된 법(re)'] = questions['추출된 법(re)'].apply(ast.literal_eval)

top_k_values = [1, 5, 10]
all_check_match_dpr_results = {k: 0 for k in top_k_values}
all_check_match_bm25_results = {k: 0 for k in top_k_values}
all_check_match_reranked_results = {k: 0 for k in top_k_values}
all_check_all_match_dpr_results = all_check_all_match_bm25_results = all_check_all_match_reranked_results = 0

# 데이터프레임에 검색 결과 저장용 열 추가
questions['dpr_result'] = None
questions['bm25_result'] = None
questions['reranking_result'] = None

for idx in tqdm(range(len(questions)), desc='Retrieving questions'):
    dpr_results, bm25_results = get_all_results(questions['민원내용'][idx], q_tokenizer, q_model, p_embs, passage['passage'].tolist(),  retriever, corpus, kiwi_tokenizer, device)
    reranked_results = rerank_with_reranker(dpr_results, bm25_results, questions['민원내용'][idx], reranker_tokenizer, reranker_model)

    process_dpr_results = process_and_combine_results(dpr_results)
    process_bm25_results = process_and_combine_results(bm25_results)
    process_reranked_results = process_and_combine_results(reranked_results)

    dpr_cleaned = [remove_parentheses(item) for item in process_dpr_results]
    bm25_cleaned = [remove_parentheses(item) for item in process_bm25_results]
    reranked_cleaned = [remove_parentheses(item) for item in process_reranked_results]
    split_laws = [split_law_and_article(item) for item in questions['추출된 법(re)'][idx]]

    
    # 처리된 결과를 데이터프레임에 저장
    questions.at[idx, 'dpr_result'] = dpr_cleaned
    questions.at[idx, 'bm25_result'] = bm25_cleaned
    questions.at[idx, 'reranking_result'] = reranked_cleaned
    

    # top-k 정확도 평가
    for k in top_k_values:
        # DPR 평가
        all_check_match_dpr_results[k] += check_match(dpr_cleaned[:k], split_laws)
        # BM25 평가
        all_check_match_bm25_results[k] += check_match(bm25_cleaned[:k], split_laws)
        # Reranker 평가
        all_check_match_reranked_results[k] += check_match(reranked_cleaned[:k], split_laws)
    
    # 전체 일치 평가 (top-10 기준)
    all_check_all_match_dpr_results += check_all_match(dpr_cleaned[:10], split_laws)
    all_check_all_match_bm25_results += check_all_match(bm25_cleaned[:10], split_laws)
    all_check_all_match_reranked_results += check_all_match(reranked_cleaned[:10], split_laws)

for k in top_k_values:
    print(f"Top-{k} DPR Accuracy: {all_check_match_dpr_results[k] / len(questions):.3f}")
    print(f"Top-{k} BM25 Accuracy: {all_check_match_bm25_results[k] / len(questions):.3f}")
    print(f"Top-{k} Reranked Accuracy: {all_check_match_reranked_results[k] / len(questions):.3f}")

print(f"All Match DPR Accuracy: {all_check_all_match_dpr_results / len(questions):.3f}")
print(f"All Match BM25 Accuracy: {all_check_all_match_bm25_results / len(questions):.3f}")
print(f"All Match Reranked Accuracy: {all_check_all_match_reranked_results / len(questions):.3f}")

questions.to_csv('./results/base_retrieval_results.csv', encoding='utf-8-sig', index=False)
