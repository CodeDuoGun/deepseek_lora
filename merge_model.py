from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import shutil

# 保证原始模型的各个文件不遗漏保存到merge_path中
def copy_files_not_in_B(A_path, B_path):
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"The directory {A_path} does not exist.")
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    # 获取路径A中所有非权重文件
    files_in_A = os.listdir(A_path)
    files_in_A = set([file for file in files_in_A if not (".bin" in file or "safetensors" in file)])

    files_in_B = set(os.listdir(B_path))

    # 找到所有A中存在但B中不存在的文件
    files_to_copy = files_in_A - files_in_B

    # 将文件或文件夹复制到B路径下
    for file in files_to_copy:
        src_path = os.path.join(A_path, file)
        dst_path = os.path.join(B_path, file)

        if os.path.isdir(src_path):
            # 复制目录及其内容
            shutil.copytree(src_path, dst_path)
        else:
            # 复制文件
            shutil.copy2(src_path, dst_path)

def merge_lora_to_base_model(model_name_or_path,adapter_name_or_path,save_path):
    # 如果文件夹不存在，就创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True,)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 加载保存的 Adapter
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map="auto",trust_remote_code=True)
    # 将 Adapter 合并到基础模型中
    merged_model = model.merge_and_unload()  # PEFT 的方法将 Adapter 权重合并到基础模型
    # 保存合并后的模型
    tokenizer.save_pretrained(save_path)
    merged_model.save_pretrained(save_path, safe_serialization=False)
    copy_files_not_in_B(model_name_or_path, save_path)
    print(f"合并后的模型已保存至: {save_path}")


if __name__ == '__main__':
    model_name_or_path = 'model/deepseek-ai/deepseek-llm-7b-chat'  # 原模型地址
    adapter_name_or_path = '/home/lixinyu/nlp/yi-6b/output/deepseek-mutil-test/'  # 微调后模型的保存地址
    save_path = 'output/deepseek-multi-1-test'

    root_dir = '/home/public/TrainerShareFolder/lxy/deepseek/config-test-output'
    # 遍历 root_dir 下面的每个子文件夹
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # 检查该文件夹是否是一个有效的子文件夹
        if os.path.isdir(folder_path):
            adapter_name_or_path = folder_path  # 当前子文件夹作为 adapter_name_or_path

            # 定义每个子文件夹的 save_path
            save_path = os.path.join(folder_path, 'merge_model')

            # 检查目标文件夹 merge_model 是否已经存在
            if os.path.exists(save_path):
                print(f"Skip folder: {folder_name}, 'merge_model' already exists.")
                continue  # 跳过该文件夹

            # 使用 try-except 来捕获 merge 操作的错误
            try:
                merge_lora_to_base_model(model_name_or_path, adapter_name_or_path, save_path)
                print(f"Successfully processed folder: {folder_name}")
            except Exception as e:
                print(f"Failed to process folder: {folder_name}. Error: {e}")
                continue  # 继续处理下一个文件夹