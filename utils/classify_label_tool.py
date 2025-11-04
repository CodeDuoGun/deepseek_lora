import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt


class ImageLabelTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("舌面图像标注工具（新增舌下3）")
        self.resize(1000, 850)

        # --- 路径显示 ---
        self.path_label = QLabel("当前图片路径：未加载")
        self.path_label.setStyleSheet("font-size:14px; color:#333; padding:5px;")
        self.path_label.setWordWrap(True)

        # --- 图片显示区（固定大小） ---
        self.image_label = QLabel("请拖入图片文件夹或点击选择")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 2px dashed #aaa; font-size: 18px; color: #555; background-color:#fafafa;"
        )
        self.image_label.setFixedSize(600, 600)
        self.image_label.setScaledContents(False)
        self.image_label.mousePressEvent = self.open_folder_dialog

        # --- 分类按钮 ---
        self.btn_tongue_up = QPushButton("舌上 (0)")
        self.btn_tongue_down = QPushButton("舌下 (3)")
        self.btn_face = QPushButton("面照 (1)")
        self.btn_other = QPushButton("其他 (2)")
        self.btn_query = QPushButton("Query")

        for btn in [self.btn_tongue_up, self.btn_tongue_down, self.btn_face, self.btn_other, self.btn_query]:
            btn.setFixedHeight(50)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    padding: 8px;
                    border-radius: 8px;
                    background-color: #f0f0f0;
                }
                QPushButton:hover {
                    background-color: #d0ebff;
                }
            """)

        # --- 导航按钮 ---
        self.btn_prev = QPushButton("⬅ 上一张")
        self.btn_next = QPushButton("下一张 ➡")
        for btn in [self.btn_prev, self.btn_next]:
            btn.setFixedHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    padding: 6px;
                    border-radius: 6px;
                    background-color: #f5f5f5;
                }
                QPushButton:hover {
                    background-color: #e8f0fe;
                }
            """)

        # --- 布局 ---
        top_layout = QVBoxLayout()
        top_layout.addWidget(self.path_label)
        top_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_tongue_up)
        btn_layout.addWidget(self.btn_tongue_down)
        btn_layout.addWidget(self.btn_face)
        btn_layout.addWidget(self.btn_other)
        btn_layout.addWidget(self.btn_query)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(nav_layout)
        self.setLayout(main_layout)

        # --- 状态 ---
        self.image_list = []
        self.current_index = 0
        self.current_dir = None
        self.current_pixmap = None

        # --- 信号绑定 ---
        self.btn_tongue_up.clicked.connect(lambda: self.move_image("0"))
        self.btn_tongue_down.clicked.connect(lambda: self.move_image("3"))
        self.btn_face.clicked.connect(lambda: self.move_image("1"))
        self.btn_other.clicked.connect(lambda: self.move_image("2"))
        self.btn_query.clicked.connect(lambda: self.move_image("query"))
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)

    # ---------------- 拖拽支持 ----------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = [url.toLocalFile() for url in event.mimeData().urls()]
        path = urls[0]
        if os.path.isdir(path):
            self.load_images_from_dir(path)
        elif os.path.isfile(path):
            self.load_images_from_dir(os.path.dirname(path))

    # ---------------- 文件夹选择 ----------------
    def open_folder_dialog(self, event=None):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.load_images_from_dir(folder)

    # ---------------- 加载图片 ----------------
    def load_images_from_dir(self, directory):
        supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.image_list = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(supported_exts)
        ]
        self.image_list.sort()
        self.current_index = 0
        self.current_dir = directory
        if not self.image_list:
            QMessageBox.warning(self, "提示", "该文件夹没有图片！")
            return
        self.show_image()

    # ---------------- 显示图片 ----------------
    def show_image(self):
        if not self.image_list:
            return
        if self.current_index < 0:
            self.current_index = 0
        if self.current_index >= len(self.image_list):
            self.image_label.setText("✅ 全部标注完成！")
            self.path_label.setText("当前图片路径：——")
            return

        image_path = self.image_list[self.current_index]
        if not os.path.exists(image_path):
            # 当前图片被移走，跳过
            self.next_image()
            return

        self.current_pixmap = QPixmap(image_path)
        scaled_pixmap = self.current_pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.path_label.setText(f"当前图片路径：{image_path}  ({self.current_index + 1}/{len(self.image_list)})")

    # ---------------- 移动图片 ----------------
    def move_image(self, label):
        if not self.image_list:
            return
        src_path = self.image_list[self.current_index]
        if not os.path.exists(src_path):
            self.next_image()
            return
        dst_dir = os.path.join(self.current_dir, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.move(src_path, dst_path)
        print(f"移动：{src_path} → {dst_path}")
        self.next_image()

    # ---------------- 上一张 ----------------
    def prev_image(self):
        if not self.image_list:
            return
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = 0
        self.show_image()

    # ---------------- 下一张 ----------------
    def next_image(self):
        if not self.image_list:
            return
        self.current_index += 1
        if self.current_index >= len(self.image_list):
            self.current_index = len(self.image_list) - 1
        self.show_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    tool = ImageLabelTool()
    tool.show()
    sys.exit(app.exec_())

