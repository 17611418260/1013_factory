import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from inference import HuBMAPDataset, Model_pred, models
from torch.utils.data import DataLoader
import torch
import numpy as np
# import pdb

# pdb.set_trace()


class ImageDisplayer:
    def __init__(self, master):
        self.master = master
        self.master.title("人体器官功能性组织切片图像分割系统")

        # 创建按钮
        self.btn_select_image = tk.Button(
            master, text="选图并预测", command=self.select_image)
        self.btn_select_image.pack()

        # 创建标签用于显示原始图片
        self.label_original_image = tk.Label(master)
        self.label_original_image.pack(side=tk.LEFT, padx=10, pady=10)

        # 创建标签用于显示处理后的图片
        self.label_processed_image = tk.Label(master)
        self.label_processed_image.pack(side=tk.RIGHT, padx=10, pady=10)

    def select_image(self):
        # 打开文件对话框选择图片
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_original_image(file_path)
            self.process_and_display_image(file_path)

    def display_original_image(self, file_path):
        # 打开图片文件
        image = Image.open(file_path)

        # 调整图片大小以适应标签
        image.thumbnail((400, 400))

        # 创建 PhotoImage 对象
        photo = ImageTk.PhotoImage(image)

        # 将图片显示在原始图片标签上
        self.label_original_image.config(image=photo)
        self.label_original_image.image = photo  # 保持对图片对象的引用

    def process_and_display_image(self, file_path):
        # 打开图片文件
        image = Image.open(file_path)

        # 提取出idx
        basename = os.path.basename(file_path)
        idx = basename.split('.')[0]

        # 预测代码
        bs = 64
        TH = 0.225
        ds = HuBMAPDataset(idx)
        dl = DataLoader(ds, bs, num_workers=0, shuffle=False, pin_memory=True)
        mp = Model_pred(models, dl)
        mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
        for p, i in iter(mp):
            mask[i.item()] = p.squeeze(-1) > TH

        # reshape tiled masks into a single mask and crop padding
        mask = mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz).\
            permute(0, 2, 1, 3).reshape(ds.n0max*ds.sz, ds.n1max*ds.sz)
        mask = mask[ds.pad0//2:-(ds.pad0-ds.pad0//2) if ds.pad0 > 0 else ds.n0max*ds.sz,
                    ds.pad1//2:-(ds.pad1-ds.pad1//2) if ds.pad1 > 0 else ds.n1max*ds.sz]

        # 调整图片大小以适应标签
        mask = mask.numpy()
        mask = Image.fromarray(np.uint8(mask*255))
        mask.thumbnail((400, 400))

        # 创建 PhotoImage 对象
        processed_photo = ImageTk.PhotoImage(mask)

        # 将处理后的图片显示在标签上
        self.label_processed_image.config(image=processed_photo)
        self.label_processed_image.image = processed_photo  # 保持对图片对象的引用


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDisplayer(root)
    root.mainloop()
