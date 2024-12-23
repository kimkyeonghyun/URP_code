import re
import pandas as pd
import argparse
import torch
import datasets
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import notebook_login


def clean_text(text):
    text = re.sub(r'\([^)]*\)', '', text) 
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'\[[^]]*\]', '', text) 
    text = re.sub(r'\<[^>]*\>', '', text)

    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'[\s]+', ' ', text)
    return text.strip()

def clean_cut_text(text):
    text = re.sub(r'\([^)]*\)', '', text) 
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'\[[^]]*\]', '', text) 
    text = re.sub(r'\<[^>]*\>', '', text)

    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'[\s]+', ' ', text)

    text = re.sub(r'[^가-힣.,?!0-9\s]+', '', text)
    return text[:1000].strip()

def remain_law(text):
    text = text.split('\n\nDepartment:')[0]
    text = re.sub(r'Law: ', '', text)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### model and dataset
    parser.add_argument('--model_type', type=str, default='EEVE-10.8b')
    parser.add_argument('--cache_dir', type=str, default='./models')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--test_ratio', type=float, default=0.05)
    parser.add_argument('--cut_text', type=str, default='False', choices=['True','False'])

    ### PEFT
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--task_type', type=str, default='CAUSAL_LM', choices=['CAUSAL_LM','QUESTION_ANS'])
    parser.add_argument('--applyQLoRA', type=str, default='True', choices=['True','False'])

    ### instruction-tuning
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--max_steps', type=int, default=1000)  # train_df: 1000, train_final: 2000
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--warmup_steps', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--instruct_dir', type=str, default='./instructs')
    
    ### etc
    parser.add_argument('--hf_login', type=str, default='False', choices=['True','False'])
    parser.add_argument('--isTest', type=str, default='False', choices=['True','False'])
    args = parser.parse_args()


    if args.hf_login == 'True':
        notebook_login()

    if args.isTest == 'True':
        args.max_steps = 50


    ### construct dataset
    train_df = pd.read_csv('./materials/민원22~24년도통합_train_1101_summarysep_no12.csv', encoding='utf-8-sig')
    train_df = train_df[['민원내용', '요약 합침']]
    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)


    ### build instruction-prompt
    def generate_prompt(ds):
        prompt_lst = []
        for i in range(len(ds)):
            text = ds['민원내용'][i]
            summary = ds['요약 합침'][i]
            
            prompt_lst.append(f'''<bos><start_of_turn>user
민원: {text}

해당 민원을 가장 잘 설명하는 법률 정보를 생성하세요.
<end_of_turn>
<start_of_turn>model
{summary}<end_of_turn><eos>''')
        return prompt_lst



    ### set lora config
    lora_config = LoraConfig(r=args.r,
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout,
                             target_modules='all-linear',
                             task_type=args.task_type)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype=torch.float16)


    
    ### load model
    if args.model_type == 'gemma-2b':
        base_model = 'google/gemma-2b-it'
    if args.model_type == 'llama3-ko-8b':
        base_model = 'beomi/Llama-3-Open-Ko-8B-Instruct-preview'
    elif args.model_type == 'EEVE-10.8b':
        base_model = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

    if args.applyQLoRA == 'True':
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=args.cache_dir)
    tokenizer.padding_side = 'right'



    ### instruction-tuning
    train_data = ds['train']
    trainer = SFTTrainer(model=model,
                         train_dataset=train_data,
                         max_seq_length=args.max_seq_length,
                         args=TrainingArguments(output_dir=args.output_dir,
                                                num_train_epochs=1,
                                                max_steps=args.max_steps,
                                                per_device_train_batch_size=1,
                                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                optim='paged_adamw_8bit',
                                                warmup_steps=args.warmup_steps,
                                                learning_rate=args.learning_rate,
                                                fp16=True,
                                                logging_steps=100,
                                                push_to_hub=False,
                                                report_to='none',),
                         peft_config=lora_config,
                         formatting_func=generate_prompt)
    trainer.train()



    ### save LoRA weight
    savename = f'{args.model_type}-sumsep-no12'
    savename += f'-{str(args.max_steps)}'
    if args.isTest == 'True':
        savename += '-test'
    
    adapter_model = f'lora_adapter/{savename}'
    trainer.model.save_pretrained(adapter_model)

    ### construct fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, cache_dir=args.cache_dir)
    model = PeftModel.from_pretrained(model, adapter_model, torch_dtype=torch.float16)
    model = model.merge_and_unload()


    model.save_pretrained(savename)