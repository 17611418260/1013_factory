import os
import glob
import shutil


def check_images(folder_path):
    # 获取文件夹下所有png文件
    png_files = glob.glob(os.path.join(folder_path, '*.png'))

    # 创建一个字典，用于存储每个id对应的文件数量
    id_count = {}

    # 遍历所有文件
    for file_path in png_files:
        # 提取文件名中的id部分
        file_name = os.path.basename(file_path)
        file_id = file_name.split('_')[0]

        # 更新id_count字典
        id_count[file_id] = id_count.get(file_id, 0) + 1

    # 检查每个id的文件数量是否为9
    for file_id, count in id_count.items():
        if count == 9:
            source_path = f'D:/Google/train_images/{file_id}.tiff'
            dest_path = f'D:/BaiduNetdiskDownload/hubmap_organ_segmentation/test_images/{file_id}.tiff'
            shutil.copy2(source_path, dest_path)


# 示例使用
folder_path = "D:\BaiduNetdiskDownload\hubmap_organ_segmentation\masks"
check_images(folder_path)
