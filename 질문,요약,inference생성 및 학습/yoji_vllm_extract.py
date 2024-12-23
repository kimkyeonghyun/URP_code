import re
import os
import time
import argparse
import traceback
import pandas as pd
import torch
import ast
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from huggingface_hub import notebook_login
from vllm import LLM, SamplingParams


def get_llama_messages(system_prompt, user_prompt):
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}]
    return messages

def get_mistral_messages(instruction):
    messages = [{'role': 'user', 'content': instruction}]
    return messages

def clean_text(text):
    text = re.sub(r'\([^)]*\)', '', text) 
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'\[[^]]*\]', '', text) 
    text = re.sub(r'\<[^>]*\>', '', text)

    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'[\s]+', ' ', text)
    return text.strip()[:1000]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='EEVE-10.8b')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--max_model_len', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.3)  # 0.5
    parser.add_argument('--top_p', type=float, default=0.6)        # 0.8
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--isTest', type=str, default='False', choices=['True','False'])
    args = parser.parse_args()



    if args.model_type == 'llama3.1-70b':
        model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
        cache_dir = '../models'
    elif args.model_type == 'qwen2-72b':
        model_name = 'Qwen/Qwen2-72B-Instruct'
        cache_dir = './models/models'
    elif args.model_type == 'EEVE-10.8b':
        model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
        cache_dir = './models/models'
    elif args.model_type == 'llama3ko-8b':
        model_name = 'beomi/Llama-3-Open-Ko-8B-Instruct-preview'
        cache_dir = './models/models'

    

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                     max_tokens=args.max_tokens)
    llm = LLM(model=model_name,
              download_dir=cache_dir,
              dtype='bfloat16',
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f'loaded model: {model_name}\n')


 
    augment = pd.read_csv('./materials/민원22~24년도통합_test_1010_law.csv', encoding='utf-8-sig')
    
    prompt_lst = []
    for i in range(len(augment)):
        if args.isTest == 'True' and i == 5: break  # for test

        minwon = augment.loc[i]['민원내용']
        instruction = f"""###예시1
- 민원내용:
건축사사무소 업을 하려면 시. 도지사에게 개설신고를 해야하는데 건축사법 개정으로 대한건축사협회에 의무 가입하여야 합니다.
[제31조의3 (건축사협회의 가입의무)
제23조제1황에 따라 건축사사무소개설신고를 한 건축사는 건축사협회 정관으로 장하는 절차에 따라 건축사합회에 가입하여야 한다.]
건축사협회 가입만 하면 되어야 하는데 시. 도건축사협회까지 동시가입을 해야하고 회비관리를 시. 도건축사회에서 하고있으며 회비를 납부해도 시군에 설치되어있는 지역회에 회비를 납부하라며 납부한 회비를 한불 합니다.
지역회가입은 의무가 아니고 선택임에도 지역회에 회비납부업무를 대한건축사협회는 위임을 한것인가? 그럼 시. 도건축사회에서 회비업무는 지역회에 위임을 했다는것인지?
건축사합회의 가입의무가 있다면 건축사협회 회원가입만 하면 되어야 한다. 회비를 2~3곳까지 내야하는지?
공정위에서는 불합리한 건축사협회 정관을 바로 잡아주시기 바랍니다.

- 민원내용 주요 질문:
지역건축사회 가입이 의무가 아닌데 왜 돈내라하는가?
회비를 2~3군데 내는건 불합리한 정관이 아닌가?

- 민원요지:
대한건축사 협회 회비 수납 관련

###예시2
- 민원내용:
건축기본법 제23조 제고향에 건축관련 민원, 설계공모 업무나 도시개발 사업등을 시행하는 경우 민간전문가를 위촉하여 해당 업무의 일부를 진행. 조정할게 할수 있다. 라고 되어 있고 광제21조제3항 제4호 민간전문가의 업무범위로 ‘건축디자인에 대한 전반적인 자문과 건축디자인 시범사업 등에 대한 기획. 설계 등‘으로 되어 있습니다.
질문고) 민간전문가는 “건축디자인에 대한 전반적인 자운“으로 민간 건축디자인에 까지 자문할 수 있는지요.
질문2) 건축디자인에 대한 전반적인 자문이란 무엇을 말하는지요?

- 민원내용 주요 질문:
민간전문가는 “건축디자인에 대한 전반적인 자문“으로 민간 건축디자인에 까지 자문할 수 있나요?
건축디자인에 대한 전반적인 자문이란 무엇을 말하나요?

- 민원요지:
민간전문가의 참여 범위

###실제민원
- 민원내용: {minwon}

###요구사항
- ###예시1, ###예시2를 참고해서 ###실제민원의 "민원내용 주요 질문"과 "민원요지"를 작성해.
- ###예시1, ###예시2의 형태를 따라서 "민원내용"을 바탕으로 "민원내용 주요 질문"을 추출해.
- "민원내용 주요 질문"은 필요에 따라서 두개 이상의 질문으로 작성해.
- "민원요지"는 "민원내용"의 핵심내용을 간락하게 작성해.

- 민원내용 주요 질문:

- 민원요지:
"""
        message = get_mistral_messages(instruction)
        message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        prompt_lst.append(message)



    outputs = llm.generate(prompt_lst, sampling_params)
    contents = [output.outputs[0].text for output in outputs]
    if args.isTest == 'True':  # for test
        augment = augment[:5]

    augment['민원요지 프롬프트'] = prompt_lst
    augment['민원요지'] = contents
    augment.to_csv('./materials/민원22~24년도통합_test_1010_law_yoji.csv', encoding='utf-8-sig', index=False)