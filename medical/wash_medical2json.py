import json
import os
from openai import OpenAI
from medical.log import logger
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import time
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

示例：
输入：该患者自述：患者于2020.11.06日因急性化脓性胆囊炎伴胆囊多发结石，来我院住院，予以胆囊CT引导下穿刺引流治疗，并予以抗炎治疗，患者症状缓解，于今日来我院住院要求行胆囊切除手术治疗，为进一步诊治入住我科。病程中，患者精神可，无腹胀腹痛，食纳可，大小便无异常，无恶心呕吐，无嗳气泛酸，无寒战、发热，无黄疸，无呕血、黑便，无腹泻、腹胀，舌质红苔薄白，脉弦。，通过望闻问切判断其病症：神志清晰，精神尚可，形体适中，语言清晰，口唇红润；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌淡红，苔白，脉弦。【证候】肝郁气滞证\n【病因】肝郁气滞是指由于肝的疏泄功能异常，疏泄不及而致气机郁滞所表现的证候。\n【中医诊断】胁痛病\n【治宜】疏肝解郁、疏肝理气\n【推荐】常见中药有贯叶金丝桃、香附、月季花等；方用乳康颗粒、朴沉化郁丸、肝郁调经膏等。
输出：
{
"chief_complaint": "2020.11.06起急性化脓性胆囊炎伴胆囊多发结石",
"current_medical_history": "患者于2020.11.06因急性化脓性胆囊炎伴胆囊多发结石入院，行胆囊CT引导下穿刺引流及抗炎治疗后症状缓解。今日为进一步诊治要求行胆囊切除术再次住院。期间精神可，无腹胀腹痛，食纳可，大小便正常，无恶心呕吐、嗳气泛酸、寒战发热、黄疸、呕血黑便、腹泻腹胀等症。",
"past_medical_history": "未提及既往疾病史",
"allergy_history": "未提及药物、食物或其他过敏情况",
"family_history": "未提及家族疾病史",
"personal_history": "未提及吸烟、饮酒、职业等个人习惯或疫苗接种信息",
"surgical_and_trauma_history": "曾行胆囊穿刺引流术",
"tongue_and_facial_diagnosis": "神志清晰，精神尚可，形体适中，语言清晰，口唇红润；皮肤正常，无斑疹。头颅形态正常，白睛无黄染；耳轮正常，无生疮；颈部对称，无青筋暴露、瘿瘤；胸部对称，虚里搏动正常；腹部平坦，无痞块；爪甲红润，双下肢无浮肿；舌质淡红，苔白，脉弦。",
"abnormal_examination_results": "舌淡红，苔白，脉弦；胆囊多发结石影像",
"provisional_diagnosis": {
    "TCM_diagnosis": "胁痛病",
    "TCM_syndrome_pattern": "肝郁气滞证",
    "western_diagnosis": "急性化脓性胆囊炎伴胆囊多发结石",
    "points_of_syndrome_diff": "肝的疏泄功能异常，气机郁滞，舌淡红，苔白，脉弦"
},
"treatment": "疏肝解郁、疏肝理气治疗。拟行胆囊切除术。",
"recommended_medications": "中药：贯叶金丝桃、香附、月季花；方剂：乳康颗粒、朴沉化郁丸、肝郁调经膏"
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

def extract_batch_emr(data:list):
    res = []
    with ThreadPoolExecutor(5) as executor:
        future_to_url = {executor.submit(extract_emr, text):text for text in data}
        for future in as_completed(future_to_url):
            text = future_to_url[future]
            try:
                data = future.result()
                res.append(data)
            except Exception as exc:
                print('%r generated an exception: %s' % (text, exc))
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
            batch_data.append(text)
        else:
            extracted_res = extract_batch_emr(batch_data)
            result.extend(extracted_res)
            batch_data = []
            time.sleep(1)
        count+=1
        if count!=0 and count % 1000==0: 
            file_path = f"medical/train_data/medical_sft_train_{count}.json"
            save_train_data(result, file_path)
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