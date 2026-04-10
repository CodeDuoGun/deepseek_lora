import os
import json

# 读取电子病历、辨证类型（norm_syndrome）、辨证要点、治疗方案、中药/方剂推荐
def prepare_data():
    data_path = "TCM_SD_train_dev/train.json"

    # 打开并读取JSON文件
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # 构建指令数据
    res = []
    for patient_case in data:
        {
            "instructions": "",
            "input":"",
            "out_put": ""
        }

    