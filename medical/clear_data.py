import json
from openai import OpenAI 
from medical.log import logger
import os
import re
from typing import List, Dict

client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY", "sk-325f79c2085d481c9a8a3e625c9a698b"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)

def convert_case_to_sample(case):
    return {
        "instruction": f"根据以下病情描述，请给出中医辨证类型、辨证要点、治法、推荐方药。\n\n{case['symptom_desc']}\n\n{case['report']}",
        "input": "",
        "output": f"""中医辨证结果:{case['tcm_results']}\n辨证要点:{case['dialectical_points']}\n治疗方案{case['treatment']}\n推荐方药:{case['drug']}""",
    }

# samples = [
#     {
#         "symptom_desc": "从【分证论治】下每个辩证结果中获取【症状】获取内容",
#         "report": "暂时为空",
#         "tcm_results": "从【分证论治】下获取多个辩证结果 每个辩证对应一条数据",
#         "dialectical_points": "从【辩证要点】获取内容",
#         "treatment_suggestion": "从【分证论治】下每个辩证结果中获取【治法】获取内容",
#         "drug": "从【分证论治】下每个辩证结果中获取【方药】获取内容",
#         "causes": "从【病因病机】获取内容",
#     }
# ]

def call_llm_api(prompt, doc):
    """"""
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen2.5-32b-instruct",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "帮我提取结构化训练样本"},
        ],
        stream=True
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )
    content = ""
    for chunk in completion:
        if chunk:
            content += chunk.choices[0].delta.content
    print(content)
    # import pdb
    # pdb.set_trace()
    content = content.strip("```json").strip("```")
    try:
        res = json.loads(content)
    except Exception as e:
        logger.error(f"{doc} err {e}")
        res = []
    return res

def split_by_sections(text: str) -> List[Dict[str, str]]:
    """
    按照《中医内科学》中“第X节 xxx”样式进行切分，返回节标题与对应内容。
    
    Returns:
        List[Dict[str, str]]: 每个节包含 'section_title' 和 'content'
    """
    # 正则匹配“第X节 xxx”，X可为中文数字或阿拉伯数字，支持空格/全角空格
    section_pattern = re.compile(r"(章节:第[一二三四五六七八九十百零〇1234567890]{1,3}节[ \u3000]*[\u4e00-\u9fa5（）()、·．.0-9a-zA-Z]+)")

    matches = list(section_pattern.finditer(text))

    sections = []

    for i, match in enumerate(matches):
        start = match.end()
        section_title = match.group(1).strip()

        end = matches[i + 1].start() if i < len(matches) - 1 else len(text)
        content = text[start:end].strip()

        sections.append({
            "section_title": section_title,
            "content": content
        })

    return sections

def request_llm():
    with open("medical_data.txt", "r", encoding='utf-8') as f:
        docs = f.read()
    docs = split_by_sections(docs)
    # import pdb
    # pdb.set_trace()
    print(len(docs))

    for doc in docs:
        prompt_v1 = f"""
# 角色：你是一位资深中医知识结构化专家，擅长从《中医内科学》教材格式的章节文本中提取结构化训练样本，用于构建中医智能诊断大模型。

# 技能：
请从以下提供的章节中，提取符合如下格式的结构化训练样本，每个辩证类型对应一条样本数据。输出格式为：
samples = [
    {{
        "symptom_desc": "提取每个【分证论治】中辩证类型下的【症状】字段内容",
        "report": "",  # 该字段暂时留空
        "tcm_results": "提取【分证论治】中该条记录所属的辩证名称，如“风寒感冒”、“风热感冒”等",
        "dialectical_points": "统一提取【辨证要点】模块的全文内容，填入所有样本中",
        "treatment_suggestion": "提取该辩证类型下的【治法】字段内容",
        "drug": "提取该辩证类型下的【方药】字段内容",
        "causes": "统一提取【病因病机】模块的全文内容，填入所有样本中"
    }},
    ...
]

# 约束
1. 分隔每个“辩证类型”的开头是 “辩证类型：·xxx”，如“辩证类型：·风寒感冒”；
2. 【辨证要点】和【病因病机】是共享信息，提取一次后用于所有样本；
3. 忽略“可用成药”、“现代研究”等说明类内容，仅提取核心辨证-治疗信息；
4. 不要遗漏任何一个分证；
5. 所有文本保留原格式（如标点），不要改写；
6. 输出为符合 Python 字典数组格式的标准结构化数据（`samples = [...]`）；
7. 直接输出json，禁止使用```json 包裹 

请开始处理以下章节内容：
{doc['content']}
"""
        prompt = f"""
# 角色：你是一位资深中医知识结构化专家，擅长从《中医内科学》教材格式的章节文本中提取结构化训练样本，用于构建中医智能诊断大模型。

# 技能：
请从以下提供的章节中，提取符合如下格式的结构化训练样本，每个辩证类型对应一条样本数据。输出格式为：
samples = [
    {{
        "symptom_desc": "提取每个【分证论治】中辩证类型下的【症状】字段内容",
        "report": "",  # 该字段暂时留空
        "tcm_results": "提取【分证论治】中该条记录所属的辩证名称，如“风寒感冒”、“风热感冒”等",
        "dialectical_points": "从【辨证要点】提取，符合该辨证类型的辨证要点，填入样本中",
        "treatment_suggestion": "提取该辩证类型下的【治法】字段内容",
        "drug": "提取该辩证类型下的【方药】字段内容，以及后面对处方的介绍内容",
    }},
    ...
]

# 约束
1. 分隔每个“辩证类型”的开头是 “辩证类型：·xxx”，如“辩证类型：·风寒感冒”；
2. 【辨证要点】是共享信息，提取一次后用于所有样本；
3. 忽略“可用成药”、“现代研究”等说明类内容，仅提取核心辨证-治疗信息；
4. 不要遗漏任何一个分证；
5. 所有文本保留原格式（如标点），不要改写；
6. 输出为符合 Python 字典数组格式的标准结构化数据（`samples = [...]`）；
7. 直接输出json，禁止使用```json 包裹 

请开始处理以下章节内容：
{doc['content']}
"""

        # 这里可以调用 LLM API 进行处理
        response = call_llm_api(prompt, doc['section_title'])
        samples.extend(response)

    with open("raw_disease_cases.json", "w") as f:
        f.write(json.dumps(samples, ensure_ascii=False) + "\n")

    # with open("tcm_liver_sft.jsonl", "w") as f:
    #     for sample in samples:
    #         f.write(json.dumps(sample, ensure_ascii=False) + "\n")

request_llm()