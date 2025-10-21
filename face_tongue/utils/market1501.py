import random
import cv2
import numpy as np
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
import os


index=0
def aug_HLS(img):
    min_v = -20
    MAX_VALUE = 20
    fImg = img.astype(np.float32)
    fImg = fImg / 255.0

    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)

    lnum = random.uniform(min_v, MAX_VALUE)
    snum = random.uniform(min_v, MAX_VALUE)
    cnum = random.uniform(min_v, MAX_VALUE)
    # print(lnum, snum, cnum)

    hlsImg[:, :, 1] = (1.0 + lnum / 100.0) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1

    hlsImg[:, :, 2] = (1.0 + snum / 100.0) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    lsImg = (lsImg * 255).astype(np.uint8)
    return lsImg
def rotate_image_and_crop(image, angle):
    """
    旋转图像，并裁剪旋转后的有效区域，裁剪尺寸为原图 / 1.5。
    
    :param image: 输入的图像（numpy 数组）
    :param angle: 旋转角度（单位：度）
    :return: 旋转并裁剪后的图像
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)  # 计算图像中心

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 旋转整个图像
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 计算裁剪区域（原图缩小 1.5 倍）
    new_w, new_h = int(w / 1), int(h / 1)
    x1, y1 = center[0] - new_w // 2, center[1] - new_h // 2
    x2, y2 = center[0] + new_w // 2, center[1] + new_h // 2

    # 确保裁剪区域不超出边界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cropped_image = rotated_image[y1:y2, x1:x2]  # 裁剪最终图像

    return cropped_image

def random_crop_and_adjust(image, image_name, min_crop_ratio=0.95, max_crop_ratio=1):
    """
    随机裁剪图像，默认保留 80% 以上的区域，防止过度裁剪。
    如果裁剪区域无效，则跳过并打印文件名。
    
    :param image: 输入图像 (numpy 数组)
    :param image_name: 图像文件名（用于错误日志）
    :param min_crop_ratio: 最小保留比例 (默认 80%)
    :param max_crop_ratio: 最大保留比例 (默认 85%)
    :return: 裁剪后的图像，如果失败返回 None
    """
    orig_h, orig_w = image.shape[:2]

    # 计算允许的裁剪范围
    min_h, max_h = int(orig_h * min_crop_ratio), int(orig_h * max_crop_ratio)
    min_w, max_w = int(orig_w * min_crop_ratio), int(orig_w * max_crop_ratio)

    try:
        # 随机选择裁剪区域
        top = np.random.randint(0, max(1, orig_h - min_h))  # 避免范围为 0
        left = np.random.randint(0, max(1, orig_w - min_w))

        bottom_limit = min(orig_h, top + max_h)
        right_limit = min(orig_w, left + max_w)

        if bottom_limit <= top + min_h or right_limit <= left + min_w:
            raise ValueError("Invalid crop dimensions")

        bottom = np.random.randint(top + min_h, bottom_limit)
        right = np.random.randint(left + min_w, right_limit)

        return image[top:bottom, left:right].copy()
    
    except ValueError:
        print(f"⚠️ 跳过裁剪：{image_name}（尺寸过小或范围错误）")
        return None  # 返回 None，表示跳过该图像

class Market1501(dataset.Dataset):
    def __init__(self, datadir, transform, dtype):
        self.epoch = None
        self.transform = transform
        self.loader = default_loader
        data_path =datadir
        if dtype == 'train':
            data_path += '/train'
        elif dtype == 'test':
            data_path += '/test'
        else:
            data_path += '/query'

        self.img_files = ['%s/%s' % (i[0].replace("\\", "/"), j) for i in os.walk(data_path) for j in i[-1] if j.lower().endswith(('jpg', 'png', 'jpeg'))]
        # self.imgs = [path for path in list_pictures(data_path)]

        self.ids=["0","1",'2','3','4','5']
        self.img_d = {}

    def __getitem__(self, index):
        # if index in self.img_d:
        #     img, target, path= self.img_d[index]
        # else:
        path = self.img_files[index]
        label_txt=os.path.basename(os.path.dirname(path))
        label_txt=label_txt.split('_')[0]
        target =self.ids.index(label_txt)
        # label_txt=os.path.basename(os.path.dirname(path))
        # target =self.ids.index(label_txt)
        img = cv2.imread(path)
        rand_v = random.randrange(3)
     
        # # # image, _ = resize_and_paste(image, 56, 56)
        if self.epoch == 0:
            # file_dir = r'/home/ps/project/SLS_det_no_2025/tiaosheng_save'
            import shutil
            shutil.rmtree(r"G:\txd\rope_predmodel_save", ignore_errors=True)
            file_dir = r"G:\txd\rope_predmodel_save"
            os.makedirs(file_dir, exist_ok=True)
            full_path_result = os.path.join(file_dir, os.path.basename(path)[:-4] + ".jpg")
            cv2.imwrite(full_path_result , img)

        img = cv2.resize(img, (64,64))
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        return img, target,path

    def __len__(self):
        return len(self.img_files)
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def id(file_path):
        id=int(os.path.basename(os.path.dirname(file_path)))
        return id