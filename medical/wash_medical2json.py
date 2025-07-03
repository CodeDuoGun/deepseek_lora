import json
import os
from openai import OpenAI
from medical.log import logger
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import time
from typing import List, Dict
load_dotenv(verbose=True)

# 使用doubao模型 测试下结构化内容提取
client = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=os.getenv("API_KEY"))
with open("medical/schema/emr_json_schema.json", "r") as fp:
    emr_response = json.loads(fp.read())
def extract_emr(text):
    try:
        prompt = """
# 角色: 医疗信息抽取助手, 基于患者描述和医生诊断结果，抽取结构化的病历.
# 技能
1. 理解并解析中文医学文本，包括症状描述、诊断结果、治疗过程。
2. 结合中西医术语，归纳病历要点，输出标准结构化字段。
3. 能识别省略信息并合理补全如“未提及”。
4. 能归类舌诊面诊、检查异常、诊断与治疗等细节字段。

# 限制
1. 遇未明确提及的历史、过敏、家族信息时，标注为“未提及...”。
2. 中西医诊断须分开列出，提取中医证型和辨证要点。
3. 严格按照json格式输出
4. 推荐方药列出常用中药及具体方剂。

示例1：

输入：患者2月前无明显诱因出现头晕、行走不稳。头晕呈头昏沉感，持续无缓解，无头痛，无恶心呕吐，无视物旋转。肢体乏力，行走不稳。2天前上述情况加重，外院行头颅CT示：1.多发腔隙性脑梗塞，部分软化灶。2.轻度脑萎缩。上述症状持续存在无好转，为求中西医结合诊治，今来我院就诊，门诊拟“脑梗死”收住入院。入院时：患者神志模糊，精神萎靡，头晕头昏，言语稍欠利，双下肢行走不稳，纳食差，自主进食困难，夜寐一般，小便失禁，大便一般。 神志模糊，精神差，形体适中，语言含糊，口唇苍白；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌淡红，苔白，脉细。, 辩证方法以及治疗方法？【证候】气虚血瘀证\n【病因】是指机体脏腑功能衰退，元气不足，无力推动血液运行，致血流不畅而成瘀所产生的一系列临床表现的证候，多因久病体虚，劳累过度，年老体衰所致。\n【中医诊断】中风病\n【治宜】化瘀通络、益气活血\n【推荐】常见中药有红景天、两面针、银杏叶等；方用脑心通丸、复方地龙胶囊、脑心通片等。
输出：
{
  "chief_complaint": "2天前头晕、行走不稳加重",
  "current_medical_history": "患者2月前无明显诱因出现头晕、行走不稳，头晕为头昏沉感，持续无缓解，无头痛、无恶心呕吐、无视物旋转，伴肢体乏力。2天前上述症状加重，外院头颅CT示：1.多发腔隙性脑梗塞，部分软化灶；2.轻度脑萎缩。症状持续存在，为求中西医结合治疗来我院门诊，门诊拟“脑梗死”收住入院。入院时患者神志模糊，精神萎靡，头晕头昏，言语稍欠利，双下肢行走不稳，纳食差，自主进食困难，夜寐一般，小便失禁，大便一般。",
  "past_medical_history": "未提及既往疾病史",
  "allergy_history": "未提及药物、食物或其他过敏情况",
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
    
        stream_res =client.chat.completions.create(
        # 替换 <MODEL> 为模型的Model ID , 查询Model ID：https://www.volcengine.com/docs/82379/1330310
            model="doubao-1.5-pro-32k-250115",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"患者描述和医生诊断结果:{text}"},
            ],
            # stream=True, 
            response_format=emr_response
        )
        logger.info(f"send resp to llm end")
        result = json.loads(stream_res.choices[0].message.content)
        logger.debug(result)
        return result
            
    except Exception:
        logger.error(f"error to deal {text} for {traceback.format_exc()}")
    return {}

def extract_batch_emr(data:List[Dict]):
    res = []
    instruction_text = "根据病史、症状、检查报告结果等医案信息，提取标准结构化病历信息，包括主诉、现病史、既往史、过敏史、家族史、婚育史、特殊时期、舌诊面诊结果、检查结果、初步诊断、治法与用药建议等内容，并以 JSON 格式输出。"
    if len(data) !=10:
        logger.warning(f"batch emr data size {len(data)}")
    logger.info(f"ready to extract emr {[case.keys()for case in data]}")
    with ThreadPoolExecutor(10) as executor:
        future_to_url = {executor.submit(extract_emr, text):ori_input for case in data for ori_input, text in case.items()}
        for future in as_completed(future_to_url):
            ori_text = future_to_url[future]
            try:
                data = future.result()
                res.append({"instruction": instruction_text, "input":ori_text, "output": data})
            except Exception as exc:
                logger.error('%r generated an exception: %s' % (ori_text, exc))
                res.append({})
    return res

def prepare_train_data(data_path="medical/data/medical_case.json"):
    with open(data_path, "r") as f:
        train_data = json.load(f)
    logger.info(train_data[0])

    result = []
    batch_size = 10
    batch_data = []
    count = 0
    for item in tqdm(train_data, total=len(train_data)):
        text = f"患者描述: {item['input']},医生诊断：{item['output']}"
        if not batch_data or len(batch_data)%batch_size!=0:
            batch_data.append({item["input"]:text})
        else:
            extracted_res = extract_batch_emr(batch_data)
            result.extend(extracted_res)
            batch_data = []
            count+=10
            time.sleep(1)
        if count!=0 and count % 100==0: 
            file_path = f"medical/train_data/medical_sft_train_{count}.json"
            save_train_data(result, file_path)
        if count == 100:
            break
        if count == 25000:
            break 

    if batch_data:
        extracted_res =  extract_batch_emr(batch_data)
        result.extend(extracted_res)
    
    file_path = f"medical/train_data/medical_sft_train_final.json"
    save_train_data(result, file_path)

    
def main():
    prepare_train_data()

def save_train_data(data, file_path):
    with open(file_path, "w") as fp:
        fp.write(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    # save_train_data([1, "你好啊"])
    main()