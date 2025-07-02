import requests
from bs4 import BeautifulSoup



# 中医内科学 目标网页的URL
neike_url = "https://www.zysj.com.cn/lilunshuji/neikexue/quanben.html"  # 替换为你要解析的网页地址
# 中医外科学 目标网页的URL
# waike_url = "https://www.zysj.com.cn/lilunshuji/waikexue/quanben.html"  # 替换为你要解析的网页地址
# 中医妇科学 目标网页的URL
# fuke_url = "https://www.zysj.com.cn/lilunshuji/fukexue/quanben.html"  # 替换为你要解析的网页地址
# 中医儿科学 目标网页的URL
# child_url = "https://www.zysj.com.cn/lilunshuji/erkexue/quanben.html"  # 替换为你要解析的网页地址
# 中医骨伤科学 目标网页的URL
# bone_url = "https://www.zysj.com.cn/lilunshuji/gushangxue/quanben.html"  # 替换为你要解析的网页地址
# 中医针灸学 
# url = "https://www.zysj.com.cn/lilunshuji/zhenjiuxue/quanben.html"  # 替换为你要解析的网页地址
# 中医推拿学 目标网页的URL
# url = "https://www.zysj.com.cn/lilunshuji/tunaxue/quanben.html"  # 替换为你要解析的网页地址
# Prescription 中医方剂学 目标网页的URL
# Prescription_url = "https://www.zysj.com.cn/lilunshuji/fangjixue/quanben.html"  # 替换为你要解析的网页地址
# 中医诊断学 目标网页的URL
# diagnosis_url = "https://www.zysj.com.cn/lilunshuji/zhenzhuangxue/quanben.html"  # 替换为你要解析的网页地址
# 中医基础理论 目标网页的URL
# url = "https://www.zysj.com.cn/lilunshuji/jichulilun/quanben.html"  
# 中医眼科 目标网页的URL 
# url = "https://www.zysj.com.cn/lilunshuji/yankexue/quanben.html"  # 替换为你要解析的网页地址   

def fetch_html(neike_url):
    # 发送HTTP请求获取网页内容
    response = requests.get(neike_url)
    res = []

    # 检查请求是否成功
    content = ""
    if response.status_code == 200:
        # 设置响应的编码（如果需要）
        response.encoding = 'utf-8'  # 假设网页内容是UTF-8编码

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找id为"content"的div元素
        content_div = soup.find('div', id='content')

        if content_div:
            # 查找div下的所有section元素
            sections = content_div.find_all('div', class_='section')
            print(f"找到{len(sections)}个section元素。")
        
            # 遍历每个section
            for section in sections:
                # 提取section中title。每个h3 作为第二个层级
                title = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                title_content = title.get_text(strip=True) if title else "无标题"
                print(f"标题内容: {title_content}")
                
                # if "章" not in title_content or "节" not in title_content:
                if "节" not in title_content:
                    continue
                if title:
                    print(f"标题: {title_content}")
                    content += f'章节:{title_content}\n\n"'
                # 提取section中的所有段落
                paragraphs = section.find_all('p')
                for p in paragraphs:
                    p_content = p.get_text(strip=True)
                    if p_content.startswith("·"):
                        content += f'\n辩证类型：{p_content}\n\n'
                    else:
                        print(f"段落: {p_content}")
                        content += p_content + "\n"
                content += "\n"
        else:
            print("未找到id为'content'的div元素。")
    else:
        print(f"请求失败，状态码：{response.status_code}")
    with open("medical_data.txt", "w", encoding='utf-8') as file:
        file.write(content)
    
    # with open("medical_data.json", "w", encoding='utf-8') as file:
    #     import json
    #     json.dump(res, file, ensure_ascii=False, indent=4)
    
fetch_html(neike_url)