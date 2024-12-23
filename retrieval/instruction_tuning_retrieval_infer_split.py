# --- 필요한 라이브러리 임포트 ---
# 각종 라이브러리와 데이터 처리, 모델 활용을 위해 필요한 모듈을 임포트
import torch
import numpy as np
import pandas as pd
import re
import argparse
from tqdm import tqdm
from ast import literal_eval
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- 법 조항을 분리하는 함수 ---
def split_passage(text):
    """
    입력된 텍스트를 법 이름과 조항으로 분리.
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

# --- 유틸리티 함수 정의 ---
# 다양한 데이터 처리, 평가, 전처리, 점수 계산 등의 유틸리티 함수들

def average_pool(last_hidden_states, attention_mask):
    """
    입력 문장 벡터를 평균 풀링으로 축약.
    """
    masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def exp_normalize(x):
    """
    입력 배열의 확률값을 정규화.
    """
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def split_law_and_article(text):
    # '제숫자조' 뒤에 의2와 같은 추가 내용을 포함하도록 정규식 수정
    match = re.match(r'(.*?)(제\d+조[\w]*)', text)
    if match:
        law_name = match.group(1).replace(' ', '').strip()  # 법 이름 부분
        article = match.group(2).strip()  # 조항 부분
        return [law_name, article]
    else:
        return [text.strip(), '']
def remove_parentheses(text):
    """
    텍스트에서 괄호와 내용을 제거.
    """
    return re.sub(r'\([^)]*\)', '', text).strip().split(' ; ')

def process_and_combine_results(results):
    """
    검색 결과를 적절히 전처리하여 결합.
    """
    combined_results = []
    for res in results:
        split_res = res.split(' : ')[0].split(' ; ')
        if '부칙' in split_res:
            combined_results.append(f"{split_res[0]} ; {split_res[1]} ; {split_res[2]}")
        else:
            combined_results.append(f"{split_res[0]} ; {split_res[1]}")
    return combined_results

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

# --- Passage 전처리 ---
def preprocess_passages(passage_data, tokenizer, max_length=512):
    """
    Passage 데이터를 전처리하여 토큰화 및 입력 형태로 변환.
    """
    passages = {'passages': [], 'passages_attention_mask': [], 'passage_ids': passage_data.index.tolist()}
    for idx in tqdm(range(len(passage_data['passage'])), desc='Preprocessing passages'):
        passage_token = tokenizer(passage_data['passage'][idx], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        passages['passages'].append(passage_token['input_ids'].squeeze(0))
        passages['passages_attention_mask'].append(passage_token['attention_mask'].squeeze(0))
    return passages

def compute_passage_embeddings(passages, model, device):
    """
    전처리된 Passage 데이터로부터 임베딩 계산.
    """
    model.eval()
    p_embs = []
    with torch.no_grad():
        for idx in tqdm(range(len(passages['passages'])), desc='Embedding passages'):
            passage = passages['passages'][idx].unsqueeze(0).to(device)
            attention_mask = passages['passages_attention_mask'][idx].unsqueeze(0).to(device)
            output = model(input_ids=passage, attention_mask=attention_mask)
            embedding = average_pool(output.last_hidden_state, attention_mask)
            p_embs.append(embedding)
    return torch.cat(p_embs, dim=0)

# --- 질의 처리 및 평가 ---
def get_all_results(questions, q_tokenizer, q_model, p_embs, passage_data, corpus, kiwi_tokenizer, retriever, device):
    """
    질의에 대해 DPR 및 BM25로 결과를 검색.
    """
    dpr_results = []
    bm25_results = []

    for question in questions:
        # DPR 처리
        question_token = q_tokenizer(question, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            question_output = q_model(input_ids=question_token['input_ids'], attention_mask=question_token['attention_mask'])
            question_embedding = average_pool(question_output.last_hidden_state, question_token['attention_mask'])
        similarity_scores = F.cosine_similarity(question_embedding.expand(p_embs.size(0), -1), p_embs, dim=1)

        # 점수와 passage를 함께 저장
        dpr_result = [(passage_data[idx], similarity_scores[idx].item()) for idx in torch.argsort(similarity_scores, descending=True)[:4]]
        dpr_results.append(dpr_result)

        # BM25 처리
        tokenized_question = kiwi_tokenizer(question)  # 쿼리를 토큰화
        bm25_scores = retriever.get_scores(tokenized_question)  # 쿼리에 대한 점수 계산
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:4]  # 상위 3개 추출
        bm25_result = [(passage_data[idx], bm25_scores[idx]) for idx in top_indices]
        bm25_results.append(bm25_result)

    # 점수 비교 및 passage만 남기기
    # DPR 전역 정렬
    global_sorted_dpr_results = sorted(
        [(passage, score) for sublist in dpr_results for passage, score in sublist],
        key=lambda x: x[1],
        reverse=True
    )
    sorted_dpr_passages = [passage for passage, _ in global_sorted_dpr_results]

    # BM25 전역 정렬
    global_sorted_bm25_results = sorted(
        [(passage, score) for sublist in bm25_results for passage, score in sublist],
        key=lambda x: x[1],
        reverse=True
    )
    sorted_bm25_passages = [passage for passage, _ in global_sorted_bm25_results]
    return sorted_dpr_passages, sorted_bm25_passages

def rerank_with_reranker(dpr_results, bm25_results, question, reranker_tokenizer, reranker_model, device, batch_size=16):
    """
    DPR 및 BM25 결과를 다시 정렬.
    """
    combined_results = list(set(dpr_results + bm25_results))
    pairs = [[question, passage] for passage in combined_results]
    
    # 토큰화 및 DataLoader 생성
    tokenized_inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    reranker_scores = []
    reranker_model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device),
            }
            logits = reranker_model(**inputs).logits.view(-1).float()
            reranker_scores.extend(logits.cpu().numpy())

    reranker_scores = exp_normalize(np.array(reranker_scores))
    sorted_results = sorted(zip(combined_results, reranker_scores), key=lambda x: x[1], reverse=True)
    return [result[0] for result in sorted_results]

def process_inference(inference):
    """'inference' 문자열을 전처리."""
    split_list = [item for item in inference.split('\n') if item.strip() != ""]
    split_split = []
    for item in split_list:
        if len(split_split) == 3:  # 최대 3개까지만 추가
            break
        split_split.append(item.strip())
    return split_split


def main():
    # --- Argument Parser ---
    # 사용자로부터 입력받을 인자 설정
    parser = argparse.ArgumentParser(description="Process question data and perform retrieval")
    parser.add_argument('--question_data', type=str, required=True, help="Path to the question_data CSV file")
    args = parser.parse_args()

    # --- 데이터 로드 및 전처리 ---
    # 질문 데이터(question_data)를 CSV 파일에서 로드
    question_data = pd.read_csv(f'./materials/{args.question_data}.csv', encoding='utf-8-sig', lineterminator='\n')
    question_data['추출된 법(re)'] = question_data['추출된 법(re)'].apply(literal_eval)  # 문자열 형태의 리스트를 실제 리스트로 변환
    
    # 불필요한 열 제거
    if 'level_0' in question_data.columns:
        question_data = question_data.drop(columns=['level_0'])
        
    # 'LLM_민원요지'열이 있는 경우 전처리
    question_data['LLM_민원요지'] = question_data['LLM_민원요지'].apply(lambda x: x.split('민원요지:')[-1].strip() if isinstance(x, str) else x)
    if 'infenrence' in question_data.columns:
        question_data['inference'] = question_data['infenrence']
    question_data = question_data[~question_data['inference'].isna()]
    question_data = question_data[question_data['추출된 법(re)'].apply(len) != 0]  # '추출된 법(re)'이 비어 있는 경우 제외
    
    # 중복 확인 및 재정렬
    if 'level_0' in question_data.columns:
        question_data = question_data.drop(columns=['level_0'])
    question_data = question_data.reset_index(drop=True)
    
    # Passage 데이터 로드 및 전처리
    passage = pd.read_csv('./law_data/law_filter_raw.csv', encoding='utf-8-sig', lineterminator='\n')
    passage[['law_name', 'law_clause']] = passage['passage'].apply(lambda x: pd.Series(split_passage(x)))  # law_name과 law_clause 분리

    # --- 모델 및 토크나이저 로드 ---
    # GPU/CPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path ='/mnt/storage1/model/'
    
    # DPR 모델 로드
    model_name = 'nlpai-lab/KoE5'
    q_tokenizer, q_model = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path), AutoModel.from_pretrained(model_name, cache_dir=cache_path).to(device)
    p_tokenizer, p_model = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path), AutoModel.from_pretrained(model_name, cache_dir=cache_path).to(device)

    # Reranker 모델 로드
    model_path = 'Dongjin-kr/ko-reranker'
    reranker_tokenizer = AutoTokenizer.from_pretrained(model_path)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=cache_path).to(device).eval()

    # Passage 데이터 전처리 및 임베딩 계산
    passages = preprocess_passages(passage, p_tokenizer)
    p_embs = compute_passage_embeddings(passages, p_model, device)

    # --- Kiwi 기반 BM25 ---
    # BM25 검색을 위한 Kiwi 토크나이저와 코퍼스 준비
    kiwi = Kiwi()
    stopwords = Stopwords()
    def kiwi_tokenizer(doc):
        tokens = kiwi.tokenize(doc, normalize_coda=True, stopwords=stopwords)
        return [token.form for token in tokens]

    # 코퍼스를 토크나이즈하여 BM25 객체 생성
    corpus = [kiwi_tokenizer(doc) for doc in tqdm(passage['passage'].tolist(), desc="Tokenizing passages with Kiwi")]
    retriever = BM25Okapi(corpus)

    # --- 최종 실행 ---
    # 질문 데이터에 대해 처리된 결과를 추가
    question_data['processed_inference'] = question_data['inference'].apply(process_inference)
    print('Question_data: ', len(question_data))

    # --- 점수 평가 ---
    # top-k 평가 및 전체 일치 평가를 위한 초기화
    top_k_values = [1, 5, 10]
    all_check_match_dpr_results = {k: 0 for k in top_k_values}
    all_check_match_bm25_results = {k: 0 for k in top_k_values}
    all_check_match_reranked_results = {k: 0 for k in top_k_values}
    all_check_all_match_dpr_results = 0
    all_check_all_match_bm25_results = 0
    all_check_all_match_reranked_results = 0

    # 결과 저장을 위한 열 추가
    question_data['dpr_result'] = None
    question_data['bm25_result'] = None
    question_data['reranking_result'] = None

    # 질문마다 검색 및 평가 수행
    for idx in tqdm(range(len(question_data)), desc='Retrieving questions'):
        # DPR 및 BM25 결과 얻기
        dpr_results, bm25_results = get_all_results(
            question_data['processed_inference'][idx], q_tokenizer, q_model, p_embs, passage['passage'].tolist(), corpus, kiwi_tokenizer, retriever, device)
        
        # Reranker를 이용한 재정렬
        reranked_results = rerank_with_reranker(
            dpr_results[:50], bm25_results[:50], question_data['LLM_민원요지'][idx], reranker_tokenizer, reranker_model, device)
        
        
        
        # 결과 전처리 [법 이름 ; 조항] 형태로 변환
        process_dpr_results = process_and_combine_results(dpr_results[:10])
        process_bm25_results = process_and_combine_results(bm25_results[:10])
        process_reranked_results = process_and_combine_results(reranked_results[:10])

        # 결과에서 괄호 제거 [[법 이름, 조항]] 형태로 변환
        dpr_cleaned = [remove_parentheses(item) for item in process_dpr_results]
        bm25_cleaned = [remove_parentheses(item) for item in process_bm25_results]
        reranked_cleaned = [remove_parentheses(item) for item in process_reranked_results]

        # 정답 데이터 전처리 [[법 이름, 조항]] 형태로 변환
        split_laws = [split_law_and_article(item) for item in question_data['추출된 법(re)'][idx]]

        # 처리된 결과를 데이터프레임에 저장
        question_data.at[idx, 'dpr_result'] = dpr_cleaned
        question_data.at[idx, 'bm25_result'] = bm25_cleaned
        question_data.at[idx, 'reranking_result'] = reranked_cleaned

        # top-k 정확도 평가
        for k in top_k_values:
            all_check_match_dpr_results[k] += check_match(dpr_cleaned[:k], split_laws)
            all_check_match_bm25_results[k] += check_match(bm25_cleaned[:k], split_laws)
            all_check_match_reranked_results[k] += check_match(reranked_cleaned[:k], split_laws)
        
        # 전체 일치 평가
        all_check_all_match_dpr_results += check_all_match(dpr_cleaned[:10], split_laws)
        all_check_all_match_bm25_results += check_all_match(bm25_cleaned[:10], split_laws)
        all_check_all_match_reranked_results += check_all_match(reranked_cleaned[:10], split_laws)

    # --- 최종 평가 결과 출력 ---
    for k in top_k_values:
        print(f"Check Match Results for top-{k} (DPR):", format(all_check_match_dpr_results[k] / len(question_data), ".3f"))
        print(f"Check Match Results for top-{k} (BM25):", format(all_check_match_bm25_results[k] / len(question_data), ".3f"))
        print(f"Check Match Results for top-{k} (Reranked):", format(all_check_match_reranked_results[k] / len(question_data), ".3f"))

    print("Check All Match Results for top-10 (DPR):", format(all_check_all_match_dpr_results / len(question_data), ".3f"))
    print("Check All Match Results for top-10 (BM25):", format(all_check_all_match_bm25_results / len(question_data), ".3f"))
    print("Check All Match Results for top-10 (Reranked):", format(all_check_all_match_reranked_results / len(question_data), ".3f"))

    # --- 결과 저장 ---
    question_data.to_csv(f'./results/{args.question_data}_resuls.csv', encoding='utf-8-sig', index=False)

if __name__ == '__main__':
    main()