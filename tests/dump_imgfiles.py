import os
import pandas as pd
import requests
import json


def download_img(save_path, img_url):
    response = requests.get(img_url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"✅ 图片已保存到：{save_path}")
        return save_path
    else:
        print(f"❌ 下载失败，HTTP状态码：{response.status_code}")
        return None


def read_csv(file="data/face_tongue_train_data/checkimgs.csv"):
    df = pd.read_csv(file)
    prefix = os.path.basename(file).split(".")[0]
    print(f"前缀：{prefix}")
    img_urls = df['inspection_report_img'].tolist()
    count = 0
    save_dir = f"data/face_tongue_train_data/{prefix}/"

    os.makedirs(save_dir, exist_ok=True)
    for img_item in img_urls:
        try:
            img_list = json.loads(img_item)
            for img_url in img_list:
                img_path = os.path.join(save_dir, f"{prefix}_{count:05d}.jpg")
                download_img(img_path, img_url=img_url["img"])
                count += 1
        except Exception as e:
            print(f"❌ 出现错误：{e}")
            continue
    return img_urls


if __name__ == "__main__":
    img_urls = read_csv()
    # download_img()