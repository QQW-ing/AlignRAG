import os
import sys
import argparse
import data_utils
import common_utils
from metrics import get_metrics
from vllm import LLM, SamplingParams
import json, argparse
from sympy import N
from tqdm import tqdm
from flashrag.config import Config
from flashrag.utils import get_generator
from data_prompt import *
from data_utils import *
from vllm import LLM, SamplingParams
import random
from openai import OpenAI
import asyncio
from openai import AsyncOpenAI
import time

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_generator(model_path, args):

    args.gpu_num = torch.cuda.device_count()

    model = LLM(model=model_path,
            seed=args.seed,
            max_model_len=32768,
            tensor_parallel_size=args.gpu_num,
            dtype='bfloat16')
    
    model_name_lower = model_path.lower()
    if "deepseek" in model_name_lower:
        print('deepseek model temperature: 0.6 and top_p: 0.95')
        temperature = 0.6
        top_p = 0.95
    elif "llama" in model_name_lower:
        print('llama model temperature: 0.6 and top_p: 0.9')
        temperature = 0.6
        top_p = 0.9
    elif "qwen" in model_name_lower:
        print('qwen model temperature: 0.7 and top_p: 0.8')
        temperature = 0.7
        top_p = 0.8
    else:
        temperature = 0.9
        top_p = 0.9
    print(f'temperature: {temperature} and top_p: {top_p}')
    sampling_params = SamplingParams(
                                temperature = temperature,
                                top_p = top_p,
                                max_tokens = 4096+1024)
    return model, sampling_params


def eval_refine(args):
    config = Config(args.config_file)

    print(f'Loading model {args.model_name_or_path}...')
    print(f'Loading critique model {args.critique_model}...')

    # 加载数据集
    data_len = []
    all_test_data = []
    q_list, psgs_list = [], []
    args.dataset_name = args.dataset_name.split(',')
    for dataset_name in args.dataset_name:
        data_path = f'evaluation/CriticRAG/dataset/{dataset_name}/test.json'
        print(f"Loading eval set from: {data_path}")
        test_data = common_utils.jload(data_path)[:args.max_instances]

        # test_data = test_data[500:510]
        all_test_data.append(test_data)
        
        for sample in test_data:
            q_list.append(sample['question'])
            psgs_list.append(build_contexts(sample, args.n_docs))
        data_len.append(len(test_data))

    # 根据不同阶段生成不同的prompt
    input_prompts = []
    if args.phase == 1:

        print("Running in phase 1 ......")

        model, sampling_params = init_generator(args.model_name_or_path, args)

        # rationale generation
        if args.n_docs > 0:
            if args.type == 'cot':
                rationale_prompt_template = RationalePromptTemplate(config)
            else:
                rationale_prompt_template = NoCoTRAGPromptTemplate(config)
        else:
            if args.type == 'cot':
                rationale_prompt_template = CoTPromptTemplate(config)
            else:
                rationale_prompt_template = NoCoTPromptTemplate(config)
                
        for query, psgs in tqdm(zip(q_list, psgs_list)):
            input_prompts.append(
                rationale_prompt_template.get_string(
                    query=query,
                    text=psgs,
                )
            )
        print("Generating rationale...")

    elif args.phase == 2:

        print("Running in phase 2 ......")

        model, sampling_params = init_generator(args.critique_model, args)

        # 加载rationale
        rationale_list = load_rationale(args.output_dir, args.dataset_name, args)
        
        # critic rationale generation
        critic_rationale_prompt_template = CritiqueRationalePromptTemplate(config)
        for query, psgs, rationale in tqdm(zip(q_list, psgs_list, rationale_list)):
            input_prompts.append(
                critic_rationale_prompt_template.get_string(
                    query=query,
                    text=psgs,
                    rationale=rationale
                )
            )
        print("Generating critic_rationale...")

    elif args.phase == 3:

        print("Running in phase 3 ......")

        model, sampling_params = init_generator(args.model_name_or_path, args)

        # 加载rationale 和 critic rationale
        rationale_list = load_rationale(args.output_dir, args.dataset_name, args)
        critic_rationale_list = load_critic_rationale(args.output_dir, args.dataset_name, args)
        
        # refine generation
        refine_prompt_template = RefinePromptTemplate(config)
        for query, psgs, rationale, critic_rationale in tqdm(zip(q_list, psgs_list, rationale_list, critic_rationale_list)):
            input_prompts.append(
                refine_prompt_template.get_string(
                    query=query,
                    text=psgs,
                    rationale=rationale,
                    critique=critic_rationale
                )
            )
        print("Generating refine...")
    
    elif args.phase == 4:

        print("Running in phase 4 ......")

        model, sampling_params = init_generator(args.model_name_or_path, args)

        # 加载rationale 和 critic rationale
        rationale_list = load_rationale(args.output_dir, args.dataset_name, args)
        critic_rationale_list = load_critic_rationale(args.output_dir, args.dataset_name, args)
        
        # rerank generation
        rerank_prompt_template = RerankRationalePromptTemplate(config)
        for query, psgs, rationale, critic_rationale in tqdm(zip(q_list, psgs_list, rationale_list, critic_rationale_list)):
            input_prompts.append(
                rerank_prompt_template.get_string(
                    query=query,
                    text=psgs,
                    rationale=rationale,
                    gold=critic_rationale
                )
            )
        print("Generating rerank...")
        
        

    print(input_prompts[0])

    # 生成结果
    start_time = time.time()
    # preds = model.generate(input_prompts, sampling_params)
    preds = model.chat(input_prompts, sampling_params)
    preds = [output.outputs[0].text for output in preds]
    print(preds[0])
    # assert len(preds) == sum(data_len)
    print(f"generation time: {time.time() - start_time} seconds")

    # 保存结果
    for i, dataset_name in enumerate(args.dataset_name):
        outputs = preds[sum(data_len[:i]):sum(data_len[:i+1])]
        print(f"Saving rationale for {dataset_name}..., {len(outputs)} examples")

        if args.phase == 1:
            output_dir = os.path.join(args.output_dir, dataset_name)
        else:
            output_dir = os.path.join(args.critique_model_path, dataset_name)
        output_file = get_output_filename(output_dir, args)

        eval_results = save_outputs(outputs, all_test_data[i], output_file, args.n_docs)
        if args.phase in [1, 3]:
            get_metrics(eval_results, output_dir, is_asqa=dataset_name == 'ASQA', sample=args.sample)

def load_rationale(output_dir, dataset_names, args):
    all_rationale_list = []
    for dataset_name in dataset_names:
        
        if args.n_docs > 0:
            if args.iters > 1:
                if args.iters == 2:
                    rationale_path = os.path.join(os.path.join(args.critique_model_path, dataset_name), f"result_refine_{args.type}.json")
                else:
                    rationale_path = os.path.join(os.path.join(args.critique_model_path, dataset_name), f"result_refine_{args.type}_iter{args.iters-1}.json")
            else:
                rationale_path = os.path.join(os.path.join(output_dir, dataset_name), f"result_rationale_{args.type}.json")
        else:
            if args.iters > 1:
                if args.iters == 2:
                    rationale_path = os.path.join(os.path.join(args.critique_model_path, dataset_name), f"result_refine_NoRAG_{args.type}.json")
                else:
                    rationale_path = os.path.join(os.path.join(args.critique_model_path, dataset_name), f"result_refine_NoRAG_{args.type}_iter{args.iters-1}.json")
            else:
                rationale_path = os.path.join(os.path.join(output_dir, dataset_name), f"result_rationale_NoRAG_{args.type}.json")
        rationale_list = common_utils.jload(rationale_path)[:args.max_instances]    
        print("rationale_list:", rationale_path)
        for data in rationale_list:
            # print(data)
            all_rationale_list.append(data['rationale'])

    return all_rationale_list

def load_critic_rationale(output_dir, dataset_names, args):
    all_critic_rationale_list = []
    for dataset_name in dataset_names:
        if args.n_docs > 0:
            if args.iters > 1:
                critic_rationale_path = os.path.join(os.path.join(args.critique_model_path, dataset_name), f"result_critic_rationale_{args.type}_iter{args.iters}.json")
            else:
                critic_rationale_path = os.path.join(os.path.join(args.critique_model_path, dataset_name), f"result_critic_rationale_{args.type}.json")
        else:
            if args.iters > 1:
                critic_rationale_path = os.path.join(os.path.join(args.critique_model_path, dataset_name), f"result_critic_rationale_NoRAG_{args.type}_iter{args.iters}.json")
            else:
                critic_rationale_path = os.path.join(os.path.join(args.critique_model_path, dataset_name), f"result_critic_rationale_NoRAG_{args.type}.json")
        critic_rationale_list = common_utils.jload(critic_rationale_path)[:args.max_instances]
        print("critic_rationale_list:", critic_rationale_path)
        for data in critic_rationale_list:
            all_critic_rationale_list.append(data['rationale'])
        
    return all_critic_rationale_list

def get_output_filename(output_dir, args):
    if args.phase == 1:
        if args.n_docs > 0:
            if args.sample:
                return os.path.join(output_dir, f"result_rationale_sample_{args.type}.json")
            elif int(args.iters) > 1:
                return os.path.join(output_dir, f"result_rationale_{args.type}_iter{args.iters}.json")
            else:
                return os.path.join(output_dir, f"result_rationale_{args.type}.json")
        else:
            if args.sample:
                return os.path.join(output_dir, f"result_rationale_sample_NoRAG_{args.type}.json")
            elif int(args.iters) > 1:
                return os.path.join(output_dir, f"result_rationale_NoRAG_{args.type}_iter{args.iters}.json")
            else:
                return os.path.join(output_dir, f"result_rationale_NoRAG_{args.type}.json")
    elif args.phase == 2:
        if args.n_docs > 0:
            if args.sample:
                return os.path.join(output_dir, f"result_critic_rationale_sample_{args.type}.json")
            elif int(args.iters) > 1:
                return os.path.join(output_dir, f"result_critic_rationale_{args.type}_iter{args.iters}.json")
            else:
                return os.path.join(output_dir, f"result_critic_rationale_{args.type}.json")
        else:
            if args.sample:
                return os.path.join(output_dir, f"result_critic_rationale_sample_NoRAG_{args.type}.json")
            elif int(args.iters) > 1:
                return os.path.join(output_dir, f"result_critic_rationale_NoRAG_{args.type}_iter{args.iters}.json")
            else:
                return os.path.join(output_dir, f"result_critic_rationale_NoRAG_{args.type}.json")
    elif args.phase == 3:
        if args.n_docs > 0:
            if args.sample:
                return os.path.join(output_dir, f"result_refine_sample_{args.type}.json")
            elif int(args.iters) > 1:
                return os.path.join(output_dir, f"result_refine_{args.type}_iter{args.iters}.json")
            else:
                return os.path.join(output_dir, f"result_refine_{args.type}.json")
        else:
            if args.sample:
                return os.path.join(output_dir, f"result_refine_sample_NoRAG_{args.type}.json")
            elif int(args.iters) > 1:
                return os.path.join(output_dir, f"result_refine_NoRAG_{args.type}_iter{args.iters}.json")
            else:
                return os.path.join(output_dir, f"result_refine_NoRAG_{args.type}.json")
        
def save_outputs(outputs, test_data, output_file, n_docs):
    # Save the outputs as a JSON file.
    output_data = []
    for i, output in enumerate(outputs):
        sample = test_data[i]
        if sample["ctxs"][0]['score'] != None and sample["ctxs"][1]['score'] != None:
            output_data.append({
                "question": sample["question"],
                "answers": sample["answers"],
                "qa_pairs": sample["qa_pairs"] if "qa_pairs" in sample else None,
                "rationale": output,
                "ctxs": sample["ctxs"][:n_docs][::-1] if (sample["ctxs"][0]['score'] > sample["ctxs"][1]['score']) else sample["ctxs"][:n_docs],
                })
        else:
            output_data.append({
                "question": sample["question"],
                "answers": sample["answers"],
                "qa_pairs": sample["qa_pairs"] if "qa_pairs" in sample else None,
                "rationale": output,
                "ctxs": sample["ctxs"][:n_docs],
                })
        
    common_utils.jdump(output_data, output_file)
    print(f"Outputs saved to {output_file}")

    return output_data

if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--rag_mode', type=str, default='SFT', help='InstructRAG model: SFT or ICL')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='name of the model in Hugging Face model hub or path to the model')
    parser.add_argument('--load_local_model', action='store_true', help='Load local model')
    parser.add_argument('--do_rationale_generation', action='store_true', help='Generate rationales on training data')
    parser.add_argument('--n_docs', type=int, default=5, help='Number of retrieved documents')
    parser.add_argument('--output_dir', type=str, help='Path to the output file')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cached models')
    parser.add_argument('--prompt_dict_path', type=str, default="src/rag.json")
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--max_tokens', type=int, default=4096+1024, help='Maximum number of tokens')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_instances', type=int, default=sys.maxsize)
    parser.add_argument('--config_file', type=str, default="src/config.json")
    parser.add_argument('--type', type=str, default='refine', help='Type of the model: refine or ensemble')
    parser.add_argument('--critique_model', type=str, default='Llama-3.1-8B-Instruct', help='Name of the critique model')
    parser.add_argument('--sample', action='store_true', help='Enable chat')
    parser.add_argument('--iters', type=int, default=1, help='Number of iterations')
    parser.add_argument('--phase', type=int, default=1, help='Number of iterations')
    parser.add_argument('--critique_model_path', type=str, default=None, help='Path to the critique model')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if not os.path.exists(args.critique_model_path):
        os.makedirs(args.critique_model_path)
    
    # if args.do_rationale_generation:
    #     generate_rationale(args)
    # else:
    #     eval_model(args)

    print("args.sample", args.sample)
    print("args.type", args.type)
    print("args.iters", args.iters)
    print("args.n_docs", args.n_docs)
    if args.type == 'cot':
        print(f"Evaluating {args.type} model...")
        eval_refine(args)
    elif args.type == 'nocot':
        print(f"Evaluating {args.type} model...")
        eval_refine(args)
    elif args.type == 'self':
        print(f"Evaluating {args.type} model...")
        eval_refine(args)
