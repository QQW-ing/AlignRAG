# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import json
import dataclasses
from tqdm import tqdm
from functools import partial
from typing import Dict, Sequence, Union

import torch
import numpy as np
import transformers
import io
from pathlib import Path
import functools

makedirs = functools.partial(os.makedirs, exist_ok=True)

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            makedirs(f_dirname)
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jdump(obj: Union[str, dict, list], f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def jdumps(obj, indent=4, default=str):
    return json.dumps(obj, indent=indent, default=default)


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"
    if question.startswith("."):
        question = question.lstrip(". ")

    return question[0].lower() + question[1:]


def build_contexts(example, n_docs):

    if len(example["ctxs"]) > 0 and example["ctxs"][0]["score"] > example["ctxs"][1]["score"]:
        ctxs_list = example["ctxs"][:n_docs][::-1]
    else:
        ctxs_list = example["ctxs"][:n_docs]

    docs_text = "\n\n".join([f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}" for idx, ctx in enumerate(ctxs_list)])
    doc_prompt = f"{docs_text}\n\n"
    
    return doc_prompt

def build_docs(example, n_docs):

    if len(example["ctxs"]) > 0 and example["ctxs"][0]["score"] > example["ctxs"][1]["score"]:
        ctxs_list = example["ctxs"][:n_docs][::-1]
    else:
        ctxs_list = example["ctxs"][:n_docs]

    docs_text = [f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}" for idx, ctx in enumerate(ctxs_list)]
    
    return docs_text

def extract_result(completion_text):
    start = completion_text.rfind('{') 
    end = completion_text.rfind('}')
    if start != -1 and end != -1 and start < end:
        content = completion_text[start + 1:end]
        if "'Score'" in content or '"Score"' in content:
            value_start = content.find(':') + 1
            value = content[value_start:].strip().strip("'\" ")
            return value
    return None



def load_data_with_passage(data_path, dataset_name):
    q, psgs, a, rationale, class_label = [], [], [], [], []
    # n_docs = 5 if dataset_name != '2WikiMultiHopQA' else 10
    n_docs = 5
    if dataset_name == 'ASQA':
        data_file = f'{data_path}/{dataset_name}/train_2k_class.json'
    else:
        data_file = f'{data_path}/{dataset_name}/train_2k_noisy.json'
    # if dataset_name == 'ASQA':
    #     data_file = f'{data_path}/{dataset_name}/train_2k_rft.json'
    # else:
    #     data_file = f'{data_path}/{dataset_name}/train_2k_rft_noisy.json'
    with open(data_file) as fr:
        lines = fr.readlines()
        for line in lines:
            line = json.loads(line)
            question = normalize_question(line['question'])
            psg = build_contexts(line, n_docs=n_docs)
            answer = line['answers']
            q.append(question)
            psgs.append(psg)
            a.append(answer)
            rationale.append(line['rationale'])
            class_label.append(line['class'])
        return q, psgs, a, rationale, class_label
    
def load_data_with_docs(data_path, dataset_name):
    q, psgs, a = [], [], []
    # n_docs = 5 if dataset_name != '2WikiMultiHopQA' else 10
    n_docs = 5  
    if dataset_name == 'ASQA':
        data_file = f'{data_path}/{dataset_name}/train_2k_class.json'
    else:
        data_file = f'{data_path}/{dataset_name}/train_2k_noisy.json'
    with open(data_file) as fr:
        lines = json.load(fr)
        for line in lines:
            question = normalize_question(line['question'])
            psg = build_docs(line, n_docs=n_docs)
            answer = line['answers']
            q.append(question)
            psgs.append(psg)
            a.append(answer)
        return q, psgs, a

