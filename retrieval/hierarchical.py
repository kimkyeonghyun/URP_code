
# --- 라이브러리 임포트 ---
import os
import sys
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import ast
import re
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def rrf_rank(dpr_results, bm25_results, k=60):
    # Create a dictionary to store the RRF scores
    rrf_scores = {}

    # Assign scores for DPR results
    for rank, doc in enumerate(dpr_results):
        doc_id = doc  # 여기서 doc 자체를 고유 식별자로 사용
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0
        rrf_scores[doc_id] += 1 / (k + rank + 1)

    # Assign scores for BM25 results
    for rank, doc in enumerate(bm25_results):
        doc_id = doc  # 여기서도 동일하게 처리
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0
        rrf_scores[doc_id] += 1 / (k + rank + 1)

    # Sort the documents by RRF score
    sorted_rrf_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Get the top results
    top_k_rrf = [doc_id for doc_id, score in sorted_rrf_scores]

    return top_k_rrf


# --- DPR 및 BM25 검색 함수 ---
def get_all_results(question: str, q_tokenizer, q_model, p_embs, passage_data, device, model_name):
    # Tokenize the question
    
    question_token = q_tokenizer(question, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    q_model.eval()

    # Compute the question embedding
    with torch.no_grad():
        if 'bert' in model_name:
            question_output = q_model(input_ids=question_token['input_ids'].to(device),
                                      attention_mask=question_token['attention_mask'].to(device),
                                      token_type_ids=question_token['token_type_ids'].to(device))
            question_embedding = question_output.pooler_output
        else:
            question_output = q_model(input_ids=question_token['input_ids'].to(device),
                                      attention_mask=question_token['attention_mask'].to(device))
            question_embedding = average_pool(question_output.last_hidden_state, question_token['attention_mask'].to(device))

    # Compute similarity scores
    question_embedding_expanded = question_embedding.expand(p_embs.size(0), -1)
    similarity_scores = F.cosine_similarity(question_embedding_expanded, p_embs, dim=1)
    rank = torch.argsort(similarity_scores, descending=True)
    # Get DPR results
    dpr_passage_results = []
    for idx in rank:
        dpr_passage_results.append(passage_data[idx.item()])
    
    def blank_tokenize(sent):
        return sent.split(' ')
    
    corpus = [blank_tokenize(doc) for doc in passage_data]
    tokenized_question = blank_tokenize(question)
    # BM25 retrieval
    retriever = BM25Okapi(corpus)
    bm25_tokenized_results = retriever.get_top_n(tokenized_question, corpus, len(corpus))
    
    # Join the tokenized BM25 results back into full sentences
    bm25_results = [' '.join(tokens) for tokens in bm25_tokenized_results]
    
    return dpr_passage_results, bm25_results


# --- 분류 모델 정의 ---
class ClassificationModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float = 0.1):
        super(ClassificationModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = output.last_hidden_state[:, 0, :]  # CLS 토큰
        return self.classifier(cls_output)

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

def remove_parentheses(text):
    """
    텍스트에서 괄호와 내용을 제거.
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

# --- 데이터 전처리 ---
def preprocess_passages(passage, tokenizer, max_seq_len, device, embedding_model):
    passages = {'passages': [], 'passages_attention_mask': []}
    if 'bert' in embedding_model:
        passages['passages_token_type_ids'] = []

    for idx in range(len(passage['passage'])):
        passage_text = passage['passage'][idx]
        passage_token = tokenizer(passage_text, padding='max_length', truncation=True, max_length=max_seq_len, return_tensors='pt')
        passages['passages'].append(passage_token['input_ids'].squeeze(0))
        passages['passages_attention_mask'].append(passage_token['attention_mask'].squeeze(0))
        if 'bert' in embedding_model:
            passages['passages_token_type_ids'].append(passage_token['token_type_ids'].squeeze(0))
    return passages


def embed_passages(passages, p_model, device, embedding_model):
    """
    Passage 데이터를 DPR 임베딩으로 변환합니다.
    """
    p_embs = []
    with torch.no_grad():
        for idx in tqdm(range(len(passages['passages'])), desc="Embedding passages"):
            passage_input_ids = passages['passages'][idx].unsqueeze(0).to(device)
            attention_mask = passages['passages_attention_mask'][idx].unsqueeze(0).to(device)
            if 'bert' in embedding_model:
                token_type_ids = passages['passages_token_type_ids'][idx].unsqueeze(0).to(device)
                output = p_model(input_ids=passage_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                output = p_model(input_ids=passage_input_ids, attention_mask=attention_mask)
            embedding = output.pooler_output if 'bert' in embedding_model else average_pool(output.last_hidden_state, attention_mask)
            p_embs.append(embedding)
    return torch.cat(p_embs, dim=0)


# --- 민원 분류 ---
def classify_complaint(complaint, model, tokenizer, law_name_list, device, max_seq_len):
    inputs = tokenizer(complaint, padding='max_length', truncation=True, max_length=max_seq_len, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        top_k_indices = torch.topk(logits, k=3, dim=1).indices.squeeze(0).tolist()
    return [law_name_list[i] for i in top_k_indices]


# --- 민원 처리 ---
def process_complaint(complaint, passages, passage_data, passage_law_names, p_embs, model, tokenizer, q_model, q_tokenizer, device, max_seq_len):
    """
    민원 데이터를 처리하여 필터링된 Passage와 DPR, BM25, RRF 결과를 반환합니다.
    """
    law_names = ['건설기술진흥법', '건설산업기본법', '건축법', '건축사법', '건축서비스산업진흥법', 
                 '건축설계공모운영지침', '경관법', '공공건축설계의도구현업무수행지침', 
                 '공공기관의정보공개에관한법률', '공공발주사업에대한건축사의업무범위와대가기준', 
                 '산업집적활성화및공장설립에관한법률', '엔지니어링산업진흥법', '주택법', '한옥등건축자산의진흥에관한법률']
    
    # 민원을 분류하여 관련 법률 필터링
    filtered_law_names = classify_complaint(complaint, model, tokenizer, law_names, device, max_seq_len)
    filtered_passages = []
    filtered_p_embs = []

    for law_name in filtered_law_names:
        indices = [i for i, law in enumerate(passage_law_names) if law == law_name]
        filtered_passages.extend([passage_data[i] for i in indices])
        filtered_p_embs.append(p_embs[indices])

    filtered_p_embs = torch.cat(filtered_p_embs, dim=0) if filtered_p_embs else torch.tensor([]).to(device)

    # 필터링된 Passage에 대해 DPR 및 BM25 검색
    dpr_results, bm25_results = get_all_results(complaint, q_tokenizer, q_model, filtered_p_embs, filtered_passages, device, model_name='e5-base')
    rrf_results = rrf_rank(dpr_results, bm25_results)

    return dpr_results, bm25_results, rrf_results


# --- 메인 실행 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
law_name_model = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(law_name_model)
model = ClassificationModel(law_name_model, num_classes=14).to(device)

#### 모델 경로 수정 필요 ####
model.load_state_dict(torch.load('./model_final/classification_model.pt', map_location=device))
model = model.eval()

# Passage 데이터 로드 및 임베딩
passage = pd.read_csv('./law_data/law_filter_raw.csv', encoding='utf-8-sig')
passage_data = passage['passage'].tolist()
passage['law_name'] = passage['passage'].apply(lambda x: x.split(' ; ')[0])
passage_law_names = passage['law_name'].tolist()

q_tokenizer, q_model = load_model_and_tokenizer(law_name_model, device)
p_tokenizer, p_model = load_model_and_tokenizer(law_name_model, device)
passages = preprocess_passages(passage, p_tokenizer, max_seq_len=512, device=device, embedding_model=law_name_model)
p_embs = embed_passages(passages, p_model, device,law_name_model)

# --- 모든 질문에 대해 처리 및 평가 ---
# 민원 데이터 로드
question_data = pd.read_csv('../민원22~24년도통합_test_1028_sepinference_final.csv', encoding='utf-8-sig')
question_data['추출된 법(re)'] = question_data['추출된 법(re)'].apply(ast.literal_eval)

# Initialize results storage for all three retrieval methods
top_k_values = [1, 5, 10]
all_check_match_dpr_results = {k: 0 for k in top_k_values}
all_check_match_bm25_results = {k: 0 for k in top_k_values}
all_check_match_reranked_results = {k: 0 for k in top_k_values}

# Initialize results storage for all_check_all_match at top-10 only
all_check_all_match_dpr_results = 0
all_check_all_match_bm25_results = 0
all_check_all_match_rrf_results = 0

# 데이터프레임에 검색 결과 저장용 열 추가
question_data['dpr_result'] = None
question_data['bm25_result'] = None
question_data['rrf_result'] = None

# Iterate over all questions in the dataset
for idx in tqdm(range(len(question_data)), desc='Retrieving questions', leave=True):
    # Extract DPR and BM25 results for each question
    dpr_results, bm25_results, rrf_results = process_complaint(question_data['민원내용'][idx], passages, passage_data, passage_law_names, p_embs, model, tokenizer, q_model, q_tokenizer, device, max_seq_len=512)

    # Process the results for comparison
    process_dpr_results = process_and_combine_results(dpr_results)
    process_bm25_results = process_and_combine_results(bm25_results)
    process_rrf_results = process_and_combine_results(rrf_results)

    dpr_cleaned = [remove_parentheses(item) for item in process_dpr_results]
    bm25_cleaned = [remove_parentheses(item) for item in process_bm25_results]
    rrf_cleaned = [remove_parentheses(item) for item in process_rrf_results]

    split_laws = [split_law_and_article(item) for item in question_data['추출된 법(re)'][idx]]

    # Check matches for top-k values (top-1, top-5, top-10)
    for k in top_k_values:
        # Apply the check_match functions for DPR results
        match_dpr_result = check_match(dpr_cleaned[:k], split_laws)
        all_check_match_dpr_results[k] += match_dpr_result

        # Apply the check_match functions for BM25 results
        match_bm25_result = check_match(bm25_cleaned[:k], split_laws)
        all_check_match_bm25_results[k] += match_bm25_result

        # Apply the check_match functions for reranked results
        match_reranked_result = check_match(rrf_cleaned[:k], split_laws)
        all_check_match_reranked_results[k] += match_reranked_result
            # 처리된 결과를 데이터프레임에 저장
        question_data.at[idx, 'dpr_result'] = dpr_cleaned
        question_data.at[idx, 'bm25_result'] = bm25_cleaned
        question_data.at[idx, 'rrf_result'] = rrf_cleaned
        

    # Only calculate all_check_all_match for top-10
    all_match_dpr_result = check_all_match(dpr_cleaned[:10], split_laws)
    all_match_bm25_result = check_all_match(bm25_cleaned[:10], split_laws)
    all_match_rrf_result = check_all_match(rrf_cleaned[:10], split_laws)

    # Accumulate all_match results for top-10 only
    all_check_all_match_dpr_results += all_match_dpr_result
    all_check_all_match_bm25_results += all_match_bm25_result
    all_check_all_match_rrf_results += all_match_rrf_result

# Print results for each top-k
for k in top_k_values:
    print(f"Check Match Results for top-{k} (DPR):", format(all_check_match_dpr_results[k] / len(question_data), ".3f"))
    print(f"Check Match Results for top-{k} (BM25):", format(all_check_match_bm25_results[k] / len(question_data), ".3f"))
    print(f"Check Match Results for top-{k} (RRF):", format(all_check_match_reranked_results[k] / len(question_data), ".3f"))

# Print all_check_all_match results for top-10 only
print("Check All Match Results for top-10 (DPR):", format(all_check_all_match_dpr_results / len(question_data), ".3f"))
print("Check All Match Results for top-10 (BM25):", format(all_check_all_match_bm25_results / len(question_data), ".3f"))
print("Check All Match Results for top-10 (RRF):", format(all_check_all_match_rrf_results / len(question_data), ".3f"))

question_data.to_csv('.results/hierarchical_retrieval_results.csv', encoding='utf-8-sig', index=False)

