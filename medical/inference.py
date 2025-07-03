from transformers import AutoTokenizer, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("medical/model/qwen3_8b_merged", device_map="auto", local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained("medical/model/qwen3_8b_merged", local_files_only=True)

# input_text = "患者女，56岁， 已婚已育育有一女，2025年2月份得着甲流之后引起咳嗽，一遇见凉气就咳嗽，或遇见热气也咳凑，到医院检查说是咳凑变异性哮喘，在检查中发现肝血管瘤和肝囊肿，因为当时咳凑只专心治疗咳凑，检查中发现肝血管瘤和肝囊肿后没有进行治疗过，在快手上发现杨主任高超的技术这样在网上挂号想让杨主任亲自诊治调理一下，让肝血管瘤和肝囊肿消掉"
# inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=1024)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("medical/model/qwen3_8b_merged", device_map="auto", local_files_only=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("medical/model/qwen3_8b_merged", local_files_only=True, trust_remote_code=True)

# ✅ 添加清晰提示，要求输出结构化 JSON
system_prompt = """你是一名专业的医疗助手，请根据以下患者描述，抽取出结构化信息，并按照如下 JSON 格式输出。若某字段无相关信息，请填写“无”。
输出格式如下：
{
  "chief_complaint": "",
  "现病史": "",
  "既往史": "",
  "过敏史": "",
  "家族史": "",
  "个人史": "",
  "手术及外伤史": "",
  "舌诊面诊": "",
  "检查异常结果": "",
  "初步诊断": {
    "中医诊断": "",
    "中医证型": "",
    "西医诊断": "",
    "辩证要点": ""
  },
  "治疗方案": "",
  "推荐方药": ""
}

请严格按照上述 JSON 格式输出，字段齐全，内容准确。
"""

user_text = "患者女，56岁，已婚已育育有一女，2025年2月份得甲流后引起咳嗽，一遇凉气就咳嗽，或遇热气也咳嗽，到医院检查为咳嗽变异性哮喘，检查中发现肝血管瘤和肝囊肿，未治疗。患者希望杨主任调理肝病，同时控制哮喘，夜间咳嗽严重影响睡眠，伴有胃食管反流，正在服用西替利嗪、孟鲁司特钠、奥美拉唑，效果不佳，情绪焦虑，希望帮助。"

input_text = system_prompt + "\n\n患者描述如下：\n" + user_text

# 推理
model.eval()
model = torch.compile(model) 
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024)

# 输出
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
