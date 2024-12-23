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
        

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='llama3.1-70b')
    parser.add_argument('--max_tokens', type=int, default=4096*2)
    parser.add_argument('--max_model_len', type=int, default=4096*4)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--isTest', type=str, default='False', choices=['True','False'])
    args = parser.parse_args()



    if args.model_type == 'llama3.1-70b':
        model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
        cache_dir = '../models'
    elif args.model_type == 'qwen2-72b':
        model_name = 'Qwen/Qwen2-72B-Instruct'
        cache_dir = './models/models'
    print(f'available devices: ', torch.cuda.device_count())



    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                     max_tokens=args.max_tokens)
    llm = LLM(model=model_name,
              download_dir=cache_dir,
              dtype='bfloat16',
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f'loaded model: {model_name}\n')



    df = pd.read_csv('./results/URP_augmented1025.csv', encoding='utf-8-sig', lineterminator='\n')
    
    prompt_lst = []
    for i in range(len(df)):
        if args.isTest == 'True' and i == 5: break  # for test
  
        law_info = df.loc[i]['law']
        
        system_prompt = '주어진 법률 정보를 두줄 이내로 요약하세요.'
        user_prompt = '주어진 법률 정보의 핵심 정보를 포함하여 두줄 이내로 요약하세요.'
        user_prompt += f' 법률 정보: {law_info}\n\n'
        user_prompt += ' 요약: ' 

        message = get_llama_messages(system_prompt, user_prompt)
        message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        prompt_lst.append(message)



    outputs = llm.generate(prompt_lst, sampling_params)
    contents = [output.outputs[0].text for output in outputs]
    if args.isTest == 'True':  # for test
        df = df[:5] 
    df['summary'] = contents


    df.to_csv('./results/URP_augmented1025_summary.csv', encoding='utf-8-sig', index=False)