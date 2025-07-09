from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

# 模型路径：合并后的模型目录
merged_model_path = "medical/output/qwen3_lora_merged"

# 加载 tokenizer（Qwen3 要加 trust_remote_code=True）
tokenizer = AutoTokenizer.from_pretrained(
    merged_model_path,
    trust_remote_code=True,
    local_files_only=True
)

# 初始化 vLLM LLM 实例
llm = LLM(
    model=merged_model_path,
    tokenizer=tokenizer,
    dtype="float16",   # Qwen3 支持 float16
    trust_remote_code=True,
    max_model_len=4096
)

# 构造 prompt（手动拼接，无“思考模式”）
content = "患者女69岁，2023年8月份查出乳腺4c,有溢液，溢液当时黑褐色，现在发淡红色，未做手术，用中药调理的，2023年7月份查出血小板低大约90.现在血小板数值49，有牙龈出血的情况，身体虚弱无力，身体过敏湿疹四个月了，乳腺4c,溢液，血小板低，牙龈出血，湿疹，已婚已育一儿一女均已婚"
prompt = """根据病史、症状、检查报告结果等医案信息，提取标准结构化病历信息，包括主诉、现病史、既往史、过敏史、家族史、婚育史、特殊时期、舌诊面诊结果、检查结果、初步诊断、治法与用药建议等内容，并以 JSON 格式输出。
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

# 拼接成 Qwen3 格式的 prompt（你也可以查 tokenizer.chat_template 实现来模拟）
prompt = f"<|im_start|>user\n{prompt}{content}<|im_end|>\n<|im_start|>assistant\n"

# 配置 sampling 参数（关闭采样、设定长度）
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024,
    top_p=1.0,
    stop=["<|im_end|>"]
)

# 执行推理
t0 = time.time()
outputs = llm.generate(prompt, sampling_params)
print(f"推理耗时: {time.time() - t0:.2f}s")

# 输出结果
response = outputs[0].outputs[0].text.strip()
print("模型输出：\n", response)
