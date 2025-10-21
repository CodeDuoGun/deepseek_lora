import glob
import os
import time
import torch
import torch.nn.functional as torch_F
import numpy as np
from addict import Dict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from addict import Dict

from typing import List

import cv2
from collections import OrderedDict
# from app.utils.tool import perf_counter_timer
from tiaosheng_new.utils.model import ft_net  # 确保 model.py 在项目路径中
import os
import cv2
from tiaosheng_new.mobilenet.mobilenetv4 import MobileNetV4

class Cls_Up_Class():
    def __init__(self, model_path,model_name="mv4", device="cuda:0"):
        if model_name=="mv4":
            self.model = self.load_mobile_net(model_path, num_classes=2).to(device)  # YOLO‑12 检测 + 跟踪
        else:
            self.model = self.load_model(model_path, num_classes=2).to(device)  # YOLO‑12 检测 + 跟踪

        self.last_log_time = time.time()
        self.frame_counter = 0

    def load_mobile_net(self, model_path, num_classes=2):

        model = MobileNetV4('MobileNetV4ConvMedium',num_classes)
        model=model.cuda()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        model.eval()
        return model
# 初始化模型
    def load_model(self,weights_path, num_classes=2):
        model = ft_net(num_classes)

        state_dict = torch.load(weights_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if "linear" in k:
                continue
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model

# 图像预处理函数
    def preprocess_image(self,img):

        img = cv2.resize(img, (64, 64))  # (512, 64)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(img).unsqueeze(0).cuda()  # Add batch dimension
        return tensor

    def preprocess_images(self, imgs):
        processed = []
        for img in imgs:
            img = cv2.resize(img, (64, 64))  # Resize to (64, 64)
            img = img.astype(np.float32)
            img = img.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
            processed.append(img)

        batch = np.stack(processed, axis=0)  # Shape: [3, 3, 64, 64]
        tensor = torch.from_numpy(batch).cuda()
        return tensor
    def predict_batch_images(self, imgs: List[np.ndarray]) -> List[int]:
        """
        imgs: List of 3 images in np.ndarray format (HWC, BGR).
        Returns: List of predicted class indices.
        """
        with torch.no_grad():
            inputs = self.preprocess_images(imgs)  # shape: [3, 3, 64, 64]
            outputs = self.model(inputs)  # shape: [3, num_classes]
            outputs = torch_F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs.data, 1)  # shape: [3]
        return preds.cpu().tolist()

    def predict_single_image(self,img: np.ndarray) -> int:
        with torch.no_grad():
            inputs = self.preprocess_image(img)
            outputs = self.model(inputs)
            outputs = torch_F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs.data, 1)
        return int(preds.item())

if __name__ == '__main__':

    model_dict = Dict({"det_path": r"tiaosheng_new/best_0721.pt", "track_buffer": 30,
                    #    "cls_tiao_mode": r"tiaosheng_new/model/0.9858_0.0000_0.0000_2.pth",
                       "cls_tiao_mode": r"models_tiaosheng_0801/mobilenet_m/0.9755_0.0000_0.0000_36.pth",
                       "match_thresh": 0.9, "min_box_area": 10, "mot20": False})

    # model_path = r"tiaosheng_new\model\0.9815_0.0000_0.0000_7.pth"
    cls_up = Cls_Up_Class(model_dict.cls_tiao_mode)
    # cls_up.model=cls_up.model.cuda()
    # root_dir = r'C:\Users\29553\Documents\xwechat_files\wxid_e9rja538jj5322_284a\msg\file\2025-06\206_9445eb279f123407ec6c45162713be7f\206_9445eb279f123407ec6c45162713be7f\0'
    # root_dir = r"C:\Users\Administrator\Pictures\demo"
    from pathlib import Path

    pred_count = {0: 0, 1: 0}
    root_dir = r"/data/data_jump/txd/train_data/0731_train/test/20250528153244686_rope_chart/0"
    pred_count = {0: 0, 1: 0}

    to_dump = True
    imgs=glob.glob(os.path.join(root_dir, "*.jpg"))

    for img_path in imgs:
            img = cv2.imread(img_path)
            pred = cls_up.predict_single_image(img)
            pred_count[pred] += 1
            print(f"Image: {img_path}, Predicted Class: {pred}")
    if to_dump:
        parent_dir = Path('/data/data_jump/txd/train_data/0731_train/test')
        # parent_dir = Path('samurai')
        print("父文件夹：", parent_dir)

        dump_dir = "err_infer"
        if os.path.exists(dump_dir):
            import shutil
            shutil.rmtree(dump_dir)
        os.makedirs(dump_dir, exist_ok=True)

        for subfolder in parent_dir.iterdir():
            if not subfolder.is_dir():
                continue
            
            for subtype_dir in subfolder.iterdir(): # subtype_dir 0 1
                err_dir_path = os.path.join(dump_dir, subfolder.name, subtype_dir.name)
                os.makedirs(err_dir_path, exist_ok=True)
                for img_path in subtype_dir.glob("*.jpg"):
                    img = cv2.imread(img_path)
                    pred = cls_up.predict_single_image(img)
                    print(f"pred res{pred} , gr res:{subtype_dir.name}")
                    if pred != subtype_dir.name:
                        # 复制错误分类的图片到错误目录
                        new_path = os.path.join(err_dir_path, img_path.name)
                        cv2.imwrite(new_path, img)
                    pred_count[pred] += 1
                    print(f"video {subfolder.name}, need_class{subtype_dir.name}, Image: {os.path.basename(img_path)}, Predicted Class: {pred}")

    print(f"Pred = 0: {pred_count[0]}")
    print(f"Pred = 1: {pred_count[1]}")

