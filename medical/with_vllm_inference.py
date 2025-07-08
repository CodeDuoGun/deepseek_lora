from openai import OpenAI

import time
# 设置本地 vLLM API 地址
client = OpenAI(base_url= "http://localhost:8000/v1", api_key="EMPTY")
# openai.api_base = "http://localhost:8000/v1"
# openai.api_key = "EMPTY"  # vLLM 默认不验证 API Key

# 模拟 prompt：关闭“思考模式”，明确结构化指令
instruction = """根据病史、症状、检查报告结果等医案信息，提取标准结构化病历信息，包括主诉、现病史、既往史、过敏史、家族史、婚育史、特殊时期、舌诊面诊结果、检查结果、初步诊断、治法与用药建议等内容，并以 JSON 格式输出。
示例：
输入：患者吃青霉素过敏，2月前无明显诱因出现头晕、行走不稳。头晕呈头昏沉感，持续无缓解，无头痛，无恶心呕吐，无视物旋转。肢体乏力，行走不稳。2天前上述情况加重，外院行头颅CT示：1.多发腔隙性脑梗塞，部分软化灶。2.轻度脑萎缩。上述症状持续存在无好转，为求中西医结合诊治，今来我院就诊，门诊拟“脑梗死”收住入院。入院时：患者神志模糊，精神萎靡，头晕头昏，言语稍欠利，双下肢行走不稳，纳食差，自主进食困难，夜寐一般，小便失禁，大便一般。 神志模糊，精神差，形体适中，语言含糊，口唇苍白；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌淡红，苔白，脉细。, 辩证方法以及治疗方法？
输出：
{
  "chief_complaint": "2天前头晕、行走不稳加重",
  "current_medical_history": "患者2月前无明显诱因出现头晕、行走不稳，头晕为头昏沉感，持续无缓解，无头痛、无恶心呕吐、无视物旋转，伴肢体乏力。2天前上述症状加重，外院头颅CT示：1.多发腔隙性脑梗塞，部分软化灶；2.轻度脑萎缩。症状持续存在，为求中西医结合治疗来我院门诊，门诊拟“脑梗死”收住入院。入院时患者神志模糊，精神萎靡，头晕头昏，言语稍欠利，双下肢行走不稳，纳食差，自主进食困难，夜寐一般，小便失禁，大便一般。",
  "past_medical_history": "未提及既往疾病史",
  "allergy_history": "青霉素过敏",
  "family_history": "未提及家族疾病史",
  "personal_history": "未提及吸烟、饮酒、职业、生育等个人史",
  "surgical_and_trauma_history": "未提及手术及外伤史",
  "marital_and_child_history": "未婚未育",
  "special_periods": "无特殊时期",
  "tongue_and_facial_diagnosis": "神志模糊，精神差，形体适中，语言含糊，口唇苍白；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌淡红，苔白，脉细。",
  "abnormal_examination_results": "头颅CT：多发腔隙性脑梗塞，部分软化灶；轻度脑萎缩",
  "provisional_diagnosis": {
    "TCM_diagnosis": "中风病",
    "TCM_syndrome_pattern": "气虚血瘀证",
    "western_diagnosis": "脑梗死，伴脑萎缩及多发腔隙性梗塞",
    "points_of_syndrome_diff": "患者年老体虚，头晕行走不稳，神志模糊，舌淡红、苔白、脉细，提示气虚为本；既往有脑梗及脑萎缩影像学表现，提示血瘀阻络，故辩证为气虚血瘀证"
  },
  "treatment": "益气活血、化瘀通络治疗",
  "recommended_medications": "中药：红景天、两面针、银杏叶；方剂：脑心通丸、复方地龙胶囊、脑心通片"
}
"""
user_content = "医案信息：患者女69岁，2023年8月份查出乳腺4c,有溢液，溢液当时黑褐色，现在发淡红色，未做手术，用中药调理的，2023年7月份查出血小板低大约90.现在血小板数值49，有牙龈出血的情况，身体虚弱无力，身体过敏湿疹四个月了，乳腺4c,溢液，血小板低，牙龈出血，湿疹，已婚已育一儿一女均已婚。"

# vLLM 的 Chat 接口消息格式（Qwen 也兼容）
messages = [
    # {"role": "system", "content": instruction},
    {"role": "user", "content": instruction+user_content}
]

start = time.perf_counter()
# 发起调用
response = client.chat.completions.create(
    model="medical/output/qwen3_lora_merged",  # 你在启动 vLLM 时注册的模型名（任意写）
    messages=messages,
    temperature=0.7,
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# 输出内容
# reply = response['choices'][0]['message']
reply = response.choices[0].message.content
print("模型输出：\n", reply)
print(f"cost {time.perf_counter() - start}")
