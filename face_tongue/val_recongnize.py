import os
import pickle
import cv2
import numpy as np
from PIL import Image
import torch
import io
from collections import defaultdict
from utils.time_str import TimeStr  # 假设你将 TimeStr 保存为 utils/time_str.py
time_str_gen = TimeStr()  # 初始化时间戳工具

from face_tongue.utils.mask_val import load_model, predict_single_image

def load_mask_from_memory(buffer):
    if isinstance(buffer, torch.Tensor):
        return buffer
    image = Image.open(io.BytesIO(buffer))
    mask = np.array(image)
    return torch.from_numpy(mask)


def draw_pic(img, points):
    prev_point = None
    for (x_val, y_val) in points:
        point = (x_val, y_val)
        if prev_point is not None:
            cv2.line(img, prev_point, point, (0, 255, 0), 1, cv2.LINE_AA)
        prev_point = point
    return img


def find_local_minima(data, step=2):
    peaks = []
    for i in range(step, len(data) - step):
        if data[i] < data[i - step] and data[i] < data[i + step]:
            if peaks and (i - 1) == peaks[-1]:
                continue  # ← 去除相邻两针跳跃点
            peaks.append(i)
    return np.array(peaks)


def draw_single_step_plot(tops,bottoms, save_path, width=64, height=64, margin=5):
    tops=np.array(tops)
    bottoms = np.array(bottoms)
    step_len = len(tops)
    # 构建横坐标（基于真实长度映射）
    x_vals = (np.arange(step_len) + margin).astype(int)
    # 归一化 y
    t_min, t_max = tops.min(), tops.max()
    b_min, b_max = bottoms.min(), bottoms.max()
    tops_y = (tops - t_min)
    bottoms_y = (bottoms - b_min) +height//2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # 画折线
    for i in range(step_len - 1):
        cv2.line(img, (x_vals[i], tops_y[i]), (x_vals[i+1], tops_y[i+1]), (0, 255, 0), 1,cv2.LINE_AA)
        cv2.line(img, (x_vals[i], bottoms_y[i]), (x_vals[i+1], bottoms_y[i+1]), (255, 0, 0), 1,cv2.LINE_AA)

    # cv2.imwrite(save_path, img)
    return img


def extract_top_bottom(video_masks, object_ids, pic_w, pic_h, total_len):
    tops, bottoms, mask_outs = [], [], []
    for frame_id, object_dict in video_masks.items():
        if frame_id > total_len:
            break
        mask_img = np.zeros((pic_w, pic_h, 3), dtype=np.uint8)
        box_found = False
        # 
        for object_id, object_item in object_dict.items():
            if object_id in object_ids:
                x1, y1, x2, y2 = object_item["box"]
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                tops.append(y1)
                bottoms.append(y2)
                box_found = True
        if not box_found:
            tops.append(tops[-1] if tops else 0)
            bottoms.append(bottoms[-1] if bottoms else 0)
        mask_outs.append(mask_img)
    return np.array(tops), np.array(bottoms), mask_outs

def assign_steps(min_indices, total_frames, fps):
    labels, steps = [], []
    current_step = 0
    gap = 0
    for frame_id in range(total_frames):
        # 是关键帧，分组
        if frame_id in min_indices:
            labels.append(0)
            steps.append(current_step)
            current_step += 1
            gap = 0
        else:
            # 不是关键帧，如果与上一个关键帧的间隔超过 fps（比如 1 秒）
            # 就视为新的跳绳动作段
            labels.append(0)
            steps.append(current_step)
            gap += 1
            if gap > fps:
                gap = 0
                current_step += 1
    return labels, steps

def group_by_step(tops, bottoms, labels, steps):
    rows = [[i, tops[i], bottoms[i], labels[i], steps[i]] for i in range(len(tops))]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[4]].append(row)
    return grouped

def main():
    # 参数配置
    model_path = r"E:\work_now\work_tiaosheng\tiaosheng_new\model\0.9911_0.0000_0.0000_30.pth"
    pkl_path = r"E:\work_now\work_tiaosheng\tiaosheng_data\tiaosheng_pkl\box6.pkl"
    save_dir = r'E:\work_now\work_tiaosheng\tiaosheng_new\tiaosheng2_0619\t2_box6666'
    os.makedirs(save_dir, exist_ok=True)
    object_ids = [0, 1, 2, 3, 4]
    pic_w, pic_h = 1800, 1800
    total_len = 10000
    fps = 24
    # 模型与数据加载
    model = load_model(model_path)
    video_masks = pickle.load(open(pkl_path, "rb"))
    if isinstance(video_masks, tuple):
        video_masks = video_masks[0]
    # 提取顶部/底部线
    tops, bottoms, mask_outs = extract_top_bottom(video_masks, object_ids, pic_w, pic_h, total_len)
    x_data = np.arange(len(tops))
    min_indices = find_local_minima(tops)
    # 分段与标签生成
    labels, steps = assign_steps(min_indices, len(tops), fps)
    # 每段保存折线图并推理
    grouped = group_by_step(tops, bottoms, labels, steps)
    tiaosheng_num = 0
    pred_count = {0: 0, 1: 0}
    for step, group in grouped.items():
        tops_ = [r[1] for r in group]
        bottoms_ = [r[2] for r in group]
        tops_ = tops_[-64:]
        bottoms_ = bottoms_[-64:]

        # 只绘图一次
        zhexian_img = draw_single_step_plot(tops_, bottoms_, save_path=None)
        # TODO: 落盘看下分类的输入结果
        # 推理
        pred = predict_single_image(zhexian_img, model) if model else 0
        pred_count[pred] += 1
        tiaosheng_num += pred
        # 创建对应子文件夹
        sub_dir = os.path.join(save_dir, str(pred))
        os.makedirs(sub_dir, exist_ok=True)
        # 保存图像
        # 生成唯一文件名
        timestamp = time_str_gen.get_time_sec()
        save_path = os.path.join(sub_dir, f"{timestamp}_step_{step:03d}.jpg")
        cv2.imwrite(save_path, zhexian_img)

    print(f"Pred = 0: {pred_count.get(0, 0)}")
    print(f"Pred = 1: {pred_count.get(1, 0)}")


if __name__ == '__main__':
    main()