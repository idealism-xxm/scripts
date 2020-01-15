import sys
from typing import List

import numpy as np
from PIL import Image

# 左上角码定位处位数
BIT_NUM = 7


def get_image_matrix(image_path: str) -> List[List[int]]:
    """读取灰度图"""
    image = Image.open(image_path)
    image = image.convert('L')
    return np.array(image).tolist()


def get_bit_length(image_matrix: List[List[int]]) -> int:
    """获取每一位所占像素个数（必须为无边框二维码）"""
    for row in image_matrix:
        # 左上角码定位处黑色部分第一行，有 7 位
        for i in range(len(row)):
            # 找到第一个白色点，计算每一位长度
            if row[i] == 255:
                return i // BIT_NUM
    raise Exception('非法图片')


def refill_bit(image_matrix: List[List[int]], bit_length: int, r: int, c, value: int):
    """用 value 填充以 (r, c) 为左上角的一位"""
    for i in range(r, r + bit_length):
        for j in range(c, c + bit_length):
            image_matrix[i][j] = value


def refill_with_0_and_255(image_matrix: List[List[int]], bit_length: int):
    """只用 0 和 255 重新填充"""
    for r in range(0, len(image_matrix), bit_length):
        for c in range(0, len(image_matrix[r]), bit_length):
            # 取中间一点的值作为填充值
            value = image_matrix[r + bit_length // 2][c + bit_length // 2]
            refill_bit(image_matrix, bit_length, r, c, value)


def resize_and_refill(image_matrix: List[List[int]], bit_length: int) -> List[List[int]]:
    """缩放图片大小，长宽能够整除 bit_length，并重新填色"""
    height = len(image_matrix) // bit_length * bit_length
    width = len(image_matrix[0]) // bit_length * bit_length
    image_matrix = image_matrix[:height]
    for i in range(height):
        image_matrix[i] = image_matrix[i][:width]

    refill_with_0_and_255(image_matrix, bit_length)
    return image_matrix


def invert_non_position_detection_pattern(image_matrix: List[List[int]], bit_length: int):
    """反色非定位处区域（非定位区域分隔位也不进行反色）"""
    height, width = len(image_matrix), len(image_matrix[0])
    size = bit_length * (BIT_NUM + 1)
    # 忽略区域列表：（左上角坐标，右下角坐标）
    position_detection_pattern_areas = [
        ((0, 0), (size - 1, size - 1)),
        ((0, width - size), (size - 1, width - 1)),
        ((height - size, 0), (height - 1, size - 1)),
    ]

    def should_ignore(r: int, c: int):
        for (lr, lc), (rr, rc) in position_detection_pattern_areas:
            if lr <= r <= rr and lc <= c <= rc:
                return True

    for i in range(height):
        for j in range(width):
            if not should_ignore(i, j):
                image_matrix[i][j] = 255 - image_matrix[i][j]


def fill_position_detection_pattern_center(image_matrix: List[List[int]], bit_length: int, value: int):
    """填充定位处中心九位"""
    height, width = len(image_matrix), len(image_matrix[0])
    size = bit_length * BIT_NUM
    # 定位区域左上角坐标列表
    position_detection_pattern_areas = [
        (0, 0),
        (0, width - size),
        (height - size, 0),
    ]
    for lr, lc in position_detection_pattern_areas:
        # 计算定位区域中心九位下标
        step = 2 * bit_length
        lr += step
        lc += step
        for r in range(lr, lr + 3 * bit_length):
            for c in range(lc, lc + 3 * bit_length):
                image_matrix[r][c] = value


def save_image(image_matrix: List[List[int]], filepath: str):
    """保存成图片到指定路径"""
    image = Image.fromarray(np.array(image_matrix, dtype=np.uint8), mode='L')
    image.save(filepath)


def invert_qrcode_context(source_path: str, target_path: str, center_value: int):
    """
    :param source_path: 待处理待二维码路径
    :param target_path: 处理后待二维码保存路径
    :param center_value: 定位区中心值
    """
    image_matrix = get_image_matrix(source_path)
    bit_length = get_bit_length(image_matrix)
    image_matrix = resize_and_refill(image_matrix, bit_length)
    invert_non_position_detection_pattern(image_matrix, bit_length)
    fill_position_detection_pattern_center(image_matrix, bit_length, center_value)
    save_image(image_matrix, target_path)


if __name__ == '__main__':
    flag_to_center_value = {
        '-e': 250,
        '-d': 0,
    }

    argv = sys.argv
    if len(argv) != 4 or argv[3] not in flag_to_center_value:
        print('Usage: python invert_qrcode_content.py {source_path} {target_path} {flag}')
        print('flag => -e for encode, -d for decode')
        exit(-1)

    source_path, target_path, flag = sys.argv[1:]
    invert_qrcode_context(source_path, target_path, flag_to_center_value[flag])
