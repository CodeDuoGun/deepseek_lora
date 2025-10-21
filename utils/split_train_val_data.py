
def split_train_val(data_dir, val_ratio=0.2):
    """
    有四个类别的数据集，类别分别为0，1，2，3
    现在你有如下目录结构的数据集：
        imgs/0/xxx.jpg
        imgs/1/xxx.jpg
    你需要将数据集按照给定的比例划分为训练集和验证集，确保每个类别的比例一致。
    划分后的目录结构如下：
        train/0/xxx.jpg
        train/1/xxx.jpg
        val/0/xxx.jpg
        val/1/xxx.jpg

    """
    pass

import os
import shutil
import random

def split_dataset(
    source_dir="imgs",
    train_dir="train",
    val_dir="test",
    split_ratio=0.8,
    seed=42
):
    random.seed(seed)

    # 获取所有类别文件夹
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    print(f"发现类别：{class_names}")

    for class_name in class_names:
        src_folder = os.path.join(source_dir, class_name)
        train_folder = os.path.join(train_dir, class_name)
        val_folder = os.path.join(val_dir, class_name)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        # 获取所有图片文件
        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(images)

        # 划分
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        print(f"类别 {class_name}: {len(train_images)} 张训练，{len(val_images)} 张验证")

        # 移动文件
        for img in train_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(train_folder, img))

        for img in val_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(val_folder, img))

    print("\n✅ 数据集划分完成！")
    print(f"训练集路径: {os.path.abspath(train_dir)}")
    print(f"验证集路径: {os.path.abspath(val_dir)}")


if __name__ == "__main__":
    # 参数可修改
    split_dataset(
        source_dir="tongue_train_imgs",   # 原始数据目录
        train_dir="tongue_train_imgs/train",   # 输出训练集
        val_dir="tongue_train_imgs/test",       # 输出验证集
        split_ratio=0.8      # 80% 训练，20% 验证
    )
