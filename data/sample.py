import json, os, argparse
import numpy as np
from tqdm import tqdm
import random
from data_utils import *
from metrics import *
import os
import pickle as pkl

def sample_train(args):
    dataset_names = ['2WikiMultiHopQA', 'ASQA', 'PopQA', 'TriviaQA', 'NaturalQuestions']
    dataset_files = [f'{args.data_path}/2WikiMultiHopQA/train.json',
                f'{args.data_path}/ASQA/train.json',
                f'{args.data_path}/PopQA/train.json',
                f'{args.data_path}/TriviaQA/train.json',
                f'{args.data_path}/NaturalQuestions/train.json']
    
    for file, dataset_name in zip(dataset_files, dataset_names):
        os.makedirs(os.path.join(args.output_path, dataset_name), exist_ok=True)
        
        with open(file) as fr:
            data = json.load(fr)
            sampled_data = data[:2000]
        
        with open(os.path.join(args.output_path, dataset_name, 'train_2k.json'), 'w') as fw:
            json.dump(sampled_data, fw)

    # os.system(f"mkdir -p {args.output_path}")


def sample_train_noise(args):
    # os.system(f"mkdir -p {args.output_path}")
    
    dataset_names = ['2WikiMultiHopQA', 'ASQA', 'PopQA', 'TriviaQA', 'NaturalQuestions']
    # dataset_files = [f'{args.data_path}/2WikiMultiHopQA/train_2k.json',
    #             f'{args.data_path}/ASQA/train_2k.json',
    #             f'{args.data_path}/PopQA/train_2k.json',
    #             f'{args.data_path}/TriviaQA/train_2k.json',
    #             f'{args.data_path}/NaturalQuestions/train_2k.json']
    
    # 使用集合存储文档以加快查找和去重
    all_psgs = set()
    all_data = []
    
    # 并行加载所有数据集的文档
    for dataset_name in tqdm(dataset_names, desc="loading data"):
        q, psgs, a, rationale = load_data_with_passage(f'{args.data_path}', dataset_name)
        all_psgs.update(psgs)
        all_data.append((q, psgs, a, rationale))
    
    # 批量处理每个数据集
    for i, (q, psgs, a, rationale) in enumerate(tqdm(all_data, desc="generating noisy data")):
        # 使用集合运算快速获取其他文档
        psgs_set = set(psgs)
        other_psgs = [psg for psg in all_psgs if psg not in psgs_set]
        
        # 为每个问题随机抽取5个其他文档作为噪声
        noisy_psgs = random.sample(other_psgs, 5)
        
        # 保存结果
        output_file = os.path.join(args.output_path, dataset_names[i], 'train_2k_noisy.json')
        with open(output_file, 'w') as fw:
            for j in range(len(q)):
                data = {'question': q[j], 'ctxs': psgs[j], 'answer': a[j], 
                        'rationale': rationale[j], 'noisy_ctxs': noisy_psgs[j]}
                fw.write(json.dumps(data) + '\n')


def sample_train_classify(args):
    dataset_names = ['2WikiMultiHopQA', 'ASQA', 'PopQA', 'TriviaQA', 'NaturalQuestions', 'hotpotqa', 'squad']
    # dataset_names = ['2WikiMultiHopQA']

    # 并行加载所有数据集的文档
    for dataset_name in tqdm(dataset_names, desc="loading data"):
        with open(f'{args.data_path}/{dataset_name}/test.json', 'r') as fr:
            test_data = json.load(fr)
            # if len(test_data) > 10000:
            #     test_data = random.sample(test_data, 1000)
        
        is_asqa = dataset_name == 'ASQA'
        accurate_len = []
        data_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
        
        for i, item in enumerate(test_data):
            psgs = build_docs(item, 5)
            num_accurate = 0
            for psg in psgs:
                eval_data = {
                    "question": item['question'],
                    "answers": item['answers'],
                    "qa_pairs": item['qa_pairs'] if 'qa_pairs' in item else None,
                    "rationale": psg,
                }

                if is_asqa:
                    rationale_str_em = compute_str_em(eval_data)
                    is_accurate = int(rationale_str_em) == 1
                else:
                    is_accurate = exact_presence(item['answers'], psg)
                num_accurate += is_accurate
            
            data_dict[num_accurate].append(i)
            accurate_len.append(num_accurate)
    
        # 统计accurate_len中0,1,2,3,4,5的个数
        accurate_len_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for i in accurate_len:
            accurate_len_count[i] += 1
        print(f"{dataset_name} accuracy distribution:", accurate_len_count)

        # 保存结果
        output_file = os.path.join(args.output_path, "test", dataset_name, 'class_test_data.pkl')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pkl.dump(data_dict, open(output_file, 'wb'))
                


def sample_test(args):
    dataset_files = [f'{args.data_path}/hotpotqa_eval.json',
                f'{args.data_path}/nq_eval.json',
                f'{args.data_path}/triviaqa_eval.json']

    os.system(f"mkdir -p {args.output_path}")

    samples = []
    for file in dataset_files:
        dataset, dataset_count = [], 0
        with open(file) as fr:    
            for line in tqdm(fr):
                dataset_count += 1
                line = json.loads(line)
                query, pos, neg, answer = line['query'], line['psgs'][:args.k], np.random.choice(line['psgs'][50:], args.k, replace=False).tolist(), line['answer']
                if "yes" in answer or 'no' in answer: continue
                which_dataset = file.split('/')[-1].split('_')[0]
                dataset.append((query, pos, neg, answer, which_dataset))
                
        
        sample_ids = np.random.choice(len(dataset), args.sample_count, replace=False) # Test: sample args.sample_count tuples on each dataset
        samples.extend(dataset[id] for id in sample_ids)
        
    with open(f'{args.output_path}/sample.json', 'w') as fw:
        for qid, data in enumerate(samples):
            fw.write(json.dumps({'qid' : qid, 'query' : data[0], 'answer':data[3], 'from':data[4]}) + '\n')

    corpus = {}
    with open(args.corpus_file) as fr:
        for line in tqdm(fr):
            line = json.loads(line)
            corpus[line['id']] = line['contents']

    with open(f'{args.output_path}/posp.json', 'w') as fw_pos, \
        open(f'{args.output_path}/negp.json', 'w') as fw_neg, \
        open(f'{args.output_path}/nsyp.json', 'w') as fw_nsy:
        
        for qid, data in tqdm(enumerate(samples)):
            
            pos_psgs = [corpus[psg_id] for psg_id in data[1]]
            neg_psgs = [corpus[psg_id] for psg_id in data[2]]
            nsy_psgs = [corpus[str(psg_id)] for psg_id in np.random.choice(args.corpus_count, args.k, replace=False)]
            
            fw_pos.write(json.dumps({'qid' : qid, 'pos_psgs' : pos_psgs}) + '\n')
            fw_neg.write(json.dumps({'qid' : qid, 'neg_psgs' : neg_psgs}) + '\n')
            fw_nsy.write(json.dumps({'qid' : qid, 'nsy_psgs' : nsy_psgs}) + '\n')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--data_path", type=str, default='data/e5/retrieve_results/')
    parser.add_argument("--output_path", type=str, default='data/e5/train')
    parser.add_argument("--corpus_file", type=str, default='FlashRAG_datasets/retrieval-corpus/wiki18_100w.jsonl')
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--corpus_count", type=int, default=21015324)
    parser.add_argument("--sample_count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    if args.mode == 'train':
        sample_train(args)
    elif args.mode == 'test':
        sample_test(args)
    elif args.mode == 'noise':
        sample_train_noise(args)
    elif args.mode == 'classify':
        sample_train_classify(args)