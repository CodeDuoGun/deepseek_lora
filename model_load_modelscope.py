from modelscope import snapshot_download
import bitsandbytes as bnb
from loguru import logger
from torch import nn
from transformers import AutoTokenizer,AutoModelForCausalLM
from modelscope.msdatasets import MsDataset
import json
from tqdm import tqdm
from datasets import load_dataset

def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names

def download_models(model_name):
    """
    需要的是魔搭社区的model
    """
    model_dir = snapshot_download(model_name, cache_dir="./model")
    return model_dir

def download_data(data_name):
    """
    魔搭社区的data
    """
    datasets = MsDataset.load(data_name,cache_dir="./data")
    return datasets

def aplaca_jsonl(train_data, output_jsonl):
    # 通过 tqdm 为循环添加进度条
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(train_data['instruction'])), desc="Saving to JSONL", unit="record"):
            instruction = train_data['instruction'][i]
            inputs = train_data['input'][i]
            outputs = train_data['output'][i]

            if instruction == None:
                instruction = ""
            if inputs == None:
                inputs = ""
            if outputs == None:
                outputs = ""

            record = {
                "instruction": instruction,
                "input": inputs,
                "output": outputs
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"数据已成功保存为 {output_jsonl}")


def extract_and_save_jsonl(input_jsonl, output_jsonl, num_records=500):
    # 读取输入的 JSONL 文件并提取前 num_records 条数据
    extracted_data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_records:  # 提取前 num_records 条数据
                break
            extracted_data.append(json.loads(line.strip()))

    # 将提取的数据保存到新的 JSONL 文件
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for record in extracted_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"成功提取并保存前 {num_records} 条数据到 {output_jsonl}")

if __name__ == '__main__':
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    download_models(model_name)








