import json, argparse, random
import numpy as np
from tqdm import tqdm
from data_utils import *


rationale_system_prompt = (
    "The following are given documents.\n\n{reference}"
)
rationale_user_prompt = "\nPlease identify documents that are useful to answer the given question: '{question}', and explain how the contents lead to the answer.\n\n"
rationale_output_prompt = "{rationale}"

critic_rationale_system_prompt = (
    "Read the following documents relevant to the given question: {query}"
    "\n{text}"
)
critic_rationale_user_prompt = (
    "\nHere is the given weak rationale: {rationale}"
    "\nPlease identify the weaknesses and hallucinations of the rationale, and give constructive criticism for improving the weak rationale."
)
critic_rationale_output_prompt = (
    "The critique for the rationale is: {critique}"
    "\n\nThe better rationale should be: {gold_rationale}"
)

critic_rationale_dpo_system_prompt = (
    "Read the following documents relevant to the given question: {query}"
    "\n{text}"
)
critic_rationale_dpo_user_prompt = (
    "\nHere is the given weak rationale: {rationale}"
    "\nPlease identify the weaknesses and hallucinations of the rationale, and give constructive criticism for improving the weak rationale."
)
critic_rationale_dpo_chosen_prompt = (
    "The critique for the rationale is: {critique_accept}"
)
critic_rationale_dpo_rejected_prompt = "{critique_reject}"



def run_rationale(args):
    q_lists, psgs_lists, rationale_lists = [], [], []
    for dataset_name in args.dataset_name:
        q_list, psgs_list, a_list, rationale_list, class_label = load_data_with_passage(args.input_path, dataset_name)
            
        q_lists.extend(q_list)
        psgs_lists.extend(psgs_list)

        rationale_list = []
        if args.do_answer:
            input_path = f'{args.input_path}/{dataset_name}/train_rationale_2k_14b.json'
        else:
            input_path = f'{args.input_path}/{dataset_name}/train_rationale_no_answer_2k_8b.json'
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                rationale = data['rationale']
                rationale_list.append(rationale)
        rationale_lists.extend(rationale_list)

    train_data = []
    for id, (q, pos_psgs, rationale) in tqdm(enumerate(zip(q_lists, psgs_lists, rationale_lists))):
        input_params = {"question": q, "reference": pos_psgs, "rationale": rationale}
        rationale_data = { 
            "instruction": rationale_system_prompt.format(**input_params),
            "input": rationale_user_prompt.format(**input_params),
            "output": rationale_output_prompt.format(**input_params)
        }
            
        train_data.append(rationale_data)
    
    random.shuffle(train_data)

    output_path = f"{args.output_file}/{args.type}_2k_14b.json"
    with open(output_path, 'w') as fw:
        json.dump(train_data,fw,indent=2)

def run_critic_rationale(args):
    q_lists, a_lists, psgs_lists, rationale_lists, critique_rationale_lists, gold_rationale_lists = [], [], [], [], [], []

    is_corrects = []
    for dataset_name in args.dataset_name:
        q_list, psgs_list, a_list, _, _ = load_data_with_passage(args.input_path, dataset_name)
            
        q_lists.extend(q_list)
        psgs_lists.extend(psgs_list)
        a_lists.extend(a_list)

        rationale_list = []
        if args.do_answer:
            input_path = f'{args.input_path}/{dataset_name}/train_rationale_2k_1b.json'
        else:
            input_path = f'{args.input_path}/{dataset_name}/train_rationale_no_answer_2k_1b.json'
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                rationale = data['rationale']
                rationale_list.append(rationale)
        rationale_lists.extend(rationale_list)
    
        critique_rationale_list = []
        input_path = f'{args.input_path}/{dataset_name}/train_{args.type}_2k_1b_distill72b_nolabel_gold.json'
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                # critiques = data['critiques'].split('</think>')[0][8:-1]
                critiques = data['critiques']
                critique_rationale_list.append(critiques)
        critique_rationale_lists.extend(critique_rationale_list)

        gold_rationale_list = []
        input_path = f'{args.input_path}/{dataset_name}/train_rationale_2k_72b.json'
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                gold_rationale_list.append(data['rationale'])
        gold_rationale_lists.extend(gold_rationale_list)

        is_corrects.extend(np.load(f'{args.input_path}/{dataset_name}/train_rationale_2k_8b_correct.npy'))

    train_data = []
    for id, (q, a, pos_psgs, rationale, critique_rationale, gold_rationale, is_correct) in tqdm(enumerate(zip(q_lists, a_lists, psgs_lists, rationale_lists, critique_rationale_lists, gold_rationale_lists, is_corrects))):
        # if is_correct:
        #     critique_rationale = "[Correct] This is a correct rationale and does not need to be improved."
        # else:
        #     critique_rationale = "[Incorrect] " + critique_rationale

        input_params = {"query": q, "text": pos_psgs, "rationale": rationale, "gold": a if args.do_answer else '', "critique": critique_rationale, "gold_rationale": gold_rationale}
        
        critic_rationale_data = { 
            "instruction": critic_rationale_system_prompt.format(**input_params),
            "input": critic_rationale_user_prompt.format(**input_params),
            "output": critic_rationale_output_prompt.format(**input_params)
        }
            
        train_data.append(critic_rationale_data)
    
    random.shuffle(train_data)
    
    output_path = f"{args.output_file}/{args.type}_2k_1b_distill72b_nolabel_gold.json"
    with open(output_path, 'w') as fw:
        json.dump(train_data,fw,indent=2)


def run_critic_rationale_dpo(args):
    q_lists, a_lists, psgs_lists, rationale_lists, critique_rationale_lists_rejects, critique_rationale_lists_accepts, gold_rationale_lists = [], [], [], [], [], [], []
    for dataset_name in args.dataset_name:
        q_list, psgs_list, a_list, rationale_list, class_label = load_data_with_passage(args.input_path, dataset_name)
            
        q_lists.extend(q_list)
        psgs_lists.extend(psgs_list)
        a_lists.extend(a_list)

        rationale_list = []
        if args.do_answer:
            input_path = f'{args.input_path}/{dataset_name}/train_rationale_2k_8b.json'
        else:
            input_path = f'{args.input_path}/{dataset_name}/train_rationale_no_answer_2k_8b.json'
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                rationale = data['rationale']
                rationale_list.append(rationale)
        rationale_lists.extend(rationale_list)
    
        critique_rationale_list_reject = []
        input_path = f'{args.input_path}/{dataset_name}/train_critic_rationale_2k_8b_distill8b_nolabel.json'
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                # critiques = data['critiques'].split('</think>')[0][8:-1]
                critiques = data['critiques']
                critique_rationale_list_reject.append(critiques)
        critique_rationale_lists_rejects.extend(critique_rationale_list_reject)

        critique_rationale_list_accept = []
        input_path = f'{args.input_path}/{dataset_name}/train_critic_rationale_2k_8b_distill72b_nolabel_gold.json'
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                # critiques = data['critiques'].split('</think>')[0][8:-1]
                critiques = data['critiques']
                critique_rationale_list_accept.append(critiques)
        critique_rationale_lists_accepts.extend(critique_rationale_list_accept)

        gold_rationale_list = []
        input_path = f'{args.input_path}/{dataset_name}/train_rationale_2k_72b.json'
        with open(input_path) as fr:
            for line in fr:
                data = json.loads(line)
                gold_rationale_list.append(data['rationale'])
        gold_rationale_lists.extend(gold_rationale_list)

    train_data = []
    for id, (q, a, pos_psgs, rationale, critique_rationale_reject, critique_rationale_accept, gold_rationale) in tqdm(enumerate(zip(q_lists, a_lists, psgs_lists, rationale_lists, critique_rationale_lists_rejects, critique_rationale_lists_accepts, gold_rationale_lists))):
        input_params = {"query": q, "text": pos_psgs, "rationale": rationale, "gold": a if args.do_answer else '', "critique_reject": critique_rationale_reject, "critique_accept": critique_rationale_accept, "gold_rationale": gold_rationale}
        critic_rationale_data = { 
            "instruction": critic_rationale_dpo_system_prompt.format(**input_params),
            "input": critic_rationale_dpo_user_prompt.format(**input_params),
            "chosen": critic_rationale_dpo_chosen_prompt.format(**input_params),
            "rejected": critic_rationale_dpo_rejected_prompt.format(**input_params)
        }
            
        train_data.append(critic_rationale_data)
    
    random.shuffle(train_data)
    
    output_path = f"{args.output_file}/{args.type}_2k_8b_distill72b_nolabel_dpo_gold.json"
    with open(output_path, 'w') as fw:
        json.dump(train_data,fw,indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_path", type=str, default='../data/e5/train/')
    parser.add_argument("--output_file", type=str, default='data/main.json')
    parser.add_argument("--position", type=str, default='random', choices=['random', 'top', 'bottom'])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--type", type=str, default='critic')
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--do_answer", type=str, default=True)
    parser.add_argument("--filter_type", type=str, default='90')
    args = parser.parse_args()

    print("Loading data...")
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    if "no_answer" in args.type:
        args.do_answer = False
    else:
        args.do_answer = True
    print("do_answer: ", args.do_answer)

    args.dataset_name = args.dataset_name.split(',')

    if args.type == 'critic_rationale' or args.type == 'critic_rationale_no_answer':
        print("Running critic_rationale...")
        run_critic_rationale(args)  
    elif args.type == 'rationale' or args.type == 'rationale_no_answer':
        print("Running rationale...")
        run_rationale(args)
    elif args.type == 'critic_rationale_dpo':
        print("Running critic_rationale_dpo...")
        run_critic_rationale_dpo(args)
    else:
        raise NotImplementedError
    
    