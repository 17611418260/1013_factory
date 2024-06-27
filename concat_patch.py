from PIL import Image
import numpy as np


def reconstruct_image(patch_prefix, output_filename):
    patches = []

    # 读取所有patch
    for i in range(9):
        patch_filename = f"D:\\BaiduNetdiskDownload\\hubmap_organ_segmentation\\masks\\{patch_prefix}{i}.png"
        patch = Image.open(patch_filename)
        patches.append(patch)

    # 获取每个patch的大小
    patch_width, patch_height = patches[0].size

    # 计算原始图片的大小
    original_width = 3 * patch_width
    original_height = 3 * patch_height

    # 创建一个新的Image对象，用于存储还原后的图片
    reconstructed_image = Image.new(
        "1", (original_width, original_height))  # 生成二值图

    # 将每个patch放回原始图片的相应位置
    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            x_offset = j * patch_width
            y_offset = i * patch_height
            # 将PngImageFile对象转换为NumPy数组，并乘以255
            patch_array = np.array(patches[index]) * 255
            # 创建Image对象，并将NumPy数组转换回图像
            patch_image = Image.fromarray(patch_array.astype(np.uint8))
            reconstructed_image.paste(patch_image, (x_offset, y_offset))

    # 保存还原后的图片
    reconstructed_image.save(output_filename)


# 使用示例
patch_prefix = "127_000"
output_filename = "reconstructed_127.png"
reconstruct_image(patch_prefix, output_filename)
