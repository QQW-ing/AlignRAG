import json, argparse
from sympy import N
from tqdm import tqdm
from flashrag.config import Config
from flashrag.utils import get_generator
from data_prompt import *
from data_utils import *
from vllm import LLM, SamplingParams
import random
def init_generator(args):
    config_dict = {"save_note": args.type,
                   "gpu_id": args.gpu_id,
                }
    config = Config(args.config_file, config_dict)

    model = LLM(model=config.generator_model_path,tokenizer=config.generator_model_path,
            seed=config.seed,
            max_model_len=32768,
            tensor_parallel_size=config.gpu_num,
            dtype='bfloat16')
    sampling_params = SamplingParams(
                                    temperature = config.generation_params['temperature'],
                                    top_k = config.generation_params['top_k'],
                                    top_p = config.generation_params['top_p'],
                                    max_tokens = config.generation_params['max_tokens'])
    return model, sampling_params

def run_rationale(args):
    config_dict = {"save_note": args.type,
                   "gpu_id": args.gpu_id,
                }
    config = Config(args.config_file, config_dict)
    print(config)
    q_prompt_template = RationalePromptTemplate(config)
    generator = get_generator(config)
    # model, sampling_params = init_generator(args)
    
    prompt_dict = jload(f'/data/rag.json')

    input_prompts = []
    data_len = []
    for dataset_name in args.dataset_name:
        q_list, psgs_list, a_list, _, class_label = load_data_with_passage(args.data_path, dataset_name)
        
        data_len.append(len(q_list))
        print(f"Generating rationale for {dataset_name}..., {len(q_list)} examples")
        for query, psgs, answer in tqdm(zip(q_list, psgs_list, a_list)):
            input_prompts.append(
                q_prompt_template.get_string(
                    query=query,
                    gold=answer, 
                    text=psgs,
                    task_specific_instruction=prompt_dict['rationale_generation_postfix_' + dataset_name]
                )
            )
    print(input_prompts[0])
    preds = generator.generate(input_prompts)
    assert len(preds) == sum(data_len)

    for i, dataset_name in enumerate(args.dataset_name):
        preds_dataset = preds[sum(data_len[:i]):sum(data_len[:i+1])]
        print(f"Saving rationale for {dataset_name}..., {len(preds_dataset)} examples")

        output_path = f'{args.output_path}/{dataset_name}/train_{args.type}_2k_70b.json'
        with open(output_path, 'w') as fw:
            for qid, rationale in enumerate(preds_dataset):
                fw.write(json.dumps({'qid':qid, 'rationale':rationale}) + '\n')

def run_critic_rationale(args):
    config_dict = {"save_note": args.type,
                   "gpu_id": args.gpu_id,
                }
    config = Config(args.config_file, config_dict)
    print(config)
    q_prompt_template = CritiqueRationalePromptTemplate(config)
    generator = get_generator(config)
    
    prompt_dict = jload(f'/data/rag.json')


    input_prompts = []
    data_len = []
    for dataset_name in args.dataset_name:
        q_list, psgs_list, a_list, _, class_label = load_data_with_passage(args.data_path, dataset_name)

        rationale_list = []
        input_path = f'{args.data_path}/{dataset_name}/train_rationale_2k_3b.json'

        print("rationale_list:", input_path)
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                rationale = data['rationale']
                rationale_list.append(rationale)

        gold_rationale_list = []
        input_path = f'{args.data_path}/{dataset_name}/train_rationale_2k_72b.json'

        print("rationale_list:", input_path)
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                rationale = data['rationale']
                gold_rationale_list.append(rationale)
        
        data_len.append(len(q_list))
        print(f"Generating critic_rationale for {dataset_name}..., {len(q_list)} examples")
        for i, (query, psgs, answer, rationale, gold_rationale) in tqdm(enumerate(zip(q_list, psgs_list, a_list, rationale_list, gold_rationale_list))):
            input_prompts.append(
                q_prompt_template.get_string(
                    query=query,
                    gold=gold_rationale, 
                    text=psgs,
                    rationale=rationale,
                    task_specific_instruction=prompt_dict['critic_' + str(class_label[i])]
                )
            )
    print(input_prompts[0])
    preds = generator.generate(input_prompts)
    assert len(preds) == sum(data_len)

    for i, dataset_name in enumerate(args.dataset_name):
        preds_dataset = preds[sum(data_len[:i]):sum(data_len[:i+1])]
        print(f"Saving critic_rationale for {dataset_name}..., {len(preds_dataset)} examples")
        output_path = f'{args.output_path}/{dataset_name}/train_{args.type}_2k_3b_distill72b_nolabel_gold.json'
        with open(output_path, 'w') as fw:
            for qid, critiques in enumerate(preds_dataset):
                fw.write(json.dumps({'qid':qid, 'critiques':critiques}) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--data_path", type=str, default='data/e5/train/')
    parser.add_argument("--output_path", type=str, default='data/e5/train/')
    parser.add_argument("--rationale_path", type=str, default='data/e5/train/')
    parser.add_argument("--critic_rationale_path", type=str, default='data/e5/train/')
    parser.add_argument("--critic_docs_path", type=str, default='data/e5/train/')
    parser.add_argument("--gpu_id", type=str)
    parser.add_argument("--config_file", type=str, default='configs/data_generation.yaml')
    parser.add_argument("--type", type=str, default='critic_rationale')
    parser.add_argument("--dataset_name", type=str)
    args = parser.parse_args()


    args.dataset_name = args.dataset_name.split(',')
    
    if args.type == 'critic_rationale':
        assert args.dataset_name is not None, "dataset_name is required for critic_rationale"
        print(f"generating critic_rationale")
        run_critic_rationale(args)
    elif args.type == 'rationale':
        assert args.dataset_name is not None, "dataset_name is required for rationale"
        print(f"generating rationale")
        run_rationale(args)

