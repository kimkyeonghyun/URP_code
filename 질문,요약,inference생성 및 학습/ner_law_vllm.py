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
from ner_samples import get_samples


def get_llama_messages(system_prompt, user_prompt):
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}]
    return messages


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='llama3.1-70b')
    parser.add_argument('--max_tokens', type=int, default=4096*2)
    parser.add_argument('--max_model_len', type=int, default=4096*4)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--isTest', type=str, default='False', choices=['True','False'])
    args = parser.parse_args()



    if args.model_type == 'llama3.1-70b':
        model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
        cache_dir = '../models'
    elif args.model_type == 'qwen2-72b':
        model_name = 'Qwen/Qwen2-72B-Instruct'
        cache_dir = './models/models'

    

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



    df = pd.read_csv('./materials/min_data_2022_2023_2024.csv', encoding='utf-8-sig')
    
    prompt_lst = []
    for i in range(len(df)):
        if args.isTest == 'True' and i == 5: break  # test
        system_prompt = 'Identify important entities in the sentences using ##Input, such as Law, Department, including the information about article and section, as in the examples below, while complying with the following conditions.'
        system_prompt += ' Create entities directly without using ##Sentence and ##Result.'
        system_prompt += ' Only laws and departments perform entity name recognition.'
        system_prompt += ' For entities that are not found, leave them blank as in the example.'
        system_prompt += " When you are finished generating queries, indicate '<|eot_id|>'"

        sample_result, sample_law  = get_samples(i%10) 
        result = df.loc[i]['처리결과']
        
        user_prompt = f'''##Sentence: {sample_result}

##Result: {sample_law}

##Input: {result}

##Result: '''

        if 'llama' in model_name or 'Qwen' in model_name:
            message = get_llama_messages(system_prompt, user_prompt)
        message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        prompt_lst.append(message)


    outputs = llm.generate(prompt_lst, sampling_params)
    contents = [output.outputs[0].text for output in outputs]
    if args.isTest == 'True':  # for test
        df = df[:5] 
    df['추출된 법'] = contents

    df.to_csv(f'./materials/min_data_2022_2023_2024_law.csv', encoding='utf-8-sig', index=False)