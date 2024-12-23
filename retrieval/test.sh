# base retrieval: DPR, BM25, ReRanker
python base_retrieval.py

# Hierrachical retrieval: 민원 내용에 대해서 classifcation 후 DPR, BM25, RRF
python hierarchical_retrieval.py # 학습된 classification model path 수정 필요

# 법 조항에 대해서 질문 생성 후 민원 요지와 유사도 계산을 통한 검색
python question_retrieval.py 

# 법 조항에 대해서 질문 요약 후 민원 요지와 유사도 계산을 통한 검색
python summary_retrieval.py 

# 최종모델: instruction tuning한 모델을 이용하여 test data에 대해 inference한 결과를 이용하여 검색
python instruction_tuning_retrieval.py