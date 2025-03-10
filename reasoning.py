import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig

### infiniAI的megrez-3b-omni测试效果
def megrez_inference(model_path, image_path):
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )
        .eval()
        .cuda()
    )

    # Chay with text
    text_messages = [
        {
            "role": "user",
            "content": {
                "text": "请问你叫什么？"
            }
        }
    ]

    # Chat with text and image
    image_messages = [
        {
            "role": "user",
            "content": {
                "text": "请你描述下图像",
                "image": image_path,
            },
        },
    ]

    MAX_NEW_TOKENS = 500

    # text
    text_response = model.chat(
        text_messages,
        sampling=False,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
    )

    image_response = model.chat(
        image_messages,
        sampling=False,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
    )

    print(text_response)
    print(image_response)


def yi_chat_model_reasoning(model_path: str, prompt: str):
    """
    单论对话的回复
    :param model_path: 模型下载地址
    :param prompt: 需要询问的问题
    :return: 回复的话
    """

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', device_map="auto")
    model.eval()

    messages = [
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    return response

def yi_base_model_reasoning(model_path: str, prompt: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=256)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def qwen_model_inference(model_path: str,prompt: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    messages = [
        {"role": "system", "content": "你是千问，一个很有用的助手"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def llama_inference(model_path: str, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map="cuda:0")

    # 将输入文本转换为模型的输入格式
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 推理过程
    with torch.no_grad():
        # 生成输出，调整参数以控制生成长度
        output = model.generate(
            inputs['input_ids'],
            max_length=2048,  # 设置最大生成长度
            num_return_sequences=1,  # 生成一个序列
            no_repeat_ngram_size=2,  # 防止重复的n-gram
            top_p=0.95,  # nucleus sampling
            temperature=0.7  # 控制输出的随机性
        )

    # 解码并打印生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def deepseek_model_inference(model_path: str, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    messages = [
        {"role": "user", "content": prompt}
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=2048)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return result

# 多轮对话推理
def deepseek_multi_conversation_inference(model_path: str, prompt: str, chat_history: list):
    """
    chat_history:[{"role": "user", "content": ……},{"role": "assistant", "content": ……}]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    messages = chat_history.copy()
    messages.append({"role": "user", "content": prompt})
    chat_history.append({"role": "user", "content": prompt})

    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=2048)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    chat_history.append({"role": "assistant", "content": result})
    return result,chat_history


if __name__ == "__main__":
    model_path = "./model/deepseek-ai/deepseek-llm-7b-chat"
    merge_path = "/home/public/TrainerShareFolder/lxy/deepseek/config-test-output/epoch-3/merge_model"

    inputs = """我最近很焦虑，我被要求参加一个节目，但是我没有任何才艺，我虽然拒绝但是也不想为难班委就答应了，现在我总觉得我会搞砸，怎么办啊？"""
    prompt = f"""现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。\n{inputs}"""
    # print(deepseek_model_inference(merge_path,inputs))

    chat_history=[]
    result, chat_history=deepseek_multi_conversation_inference(merge_path,inputs,chat_history)
    print(result)

    inputs = """可是我是个社恐，做不到怎么办？"""
    result, chat_history = deepseek_multi_conversation_inference(merge_path, inputs, chat_history)
    print(result)

    inputs = """我在数学方面总是比他落后很多，我尝试了很多方法提高，但还是觉得力不从心。"""
    result, chat_history = deepseek_multi_conversation_inference(merge_path, inputs, chat_history)
    print(result)






