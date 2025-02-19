import cv2 as cv
import numpy as np
import random
import os
from functools import partial
from filling import compensate
from cut import get_new_gt2, collect_edges

HEIGHT = 400
WIDTH = 400
final_lon = 256


# 示例模型函数，模拟不同层级联模型的输出
def optical_distortion(x, k1, k2, k3, k4):
    """光学畸变模型"""
    rd_2 = np.power(x, 2)
    rd_4 = np.power(rd_2, 2)
    rd_6 = np.power(rd_2, 3)
    rd_8 = np.power(rd_2, 4)
    return x * (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8)


def geometric_distortion(x, factor=0.2):
    """几何畸变模型"""
    return x + factor * np.power(x, 3)


def operation_distortion(x):
    """操作畸变模型"""
    return x * np.log(x + 1)


def overlay_distortion(x):
    """覆盖畸变模型"""
    return x - 0.2 * np.cos(x)


# 定义级联模型融合函数
def cascaded_model_fusion(rd, models, betas):
    """
    rd: 输入数据
    models: 模型列表，每个模型是一个函数
    betas: 权重列表，对应每个模型的权重
    """
    N = len(models)
    assert len(betas) == N, "模型数量和权重数量不匹配"

    ru = rd  # 初始输入数据
    for n in range(N):
        Gn_output = models[n](ru)
        ru = betas[n] * Gn_output + (1 - betas[n]) * ru  # 混合当前输出和前一输出
    return ru


def create_fish(srcImg, k1, k2, k3, k4):
    up = []
    down = []
    left = []
    right = []
    dstImg = np.zeros([HEIGHT, WIDTH, 3], np.uint8) + 255
    det_uv = np.zeros([HEIGHT, WIDTH, 2], np.int32) + 500

    x0 = (WIDTH - 1) / 2.
    y0 = (HEIGHT - 1) / 2.

    min_x = 9999
    cut_r = 0
    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            x_d = j - x0
            y_d = i - y0
            rd_2 = np.power(x_d, 2) + np.power(y_d, 2)
            rd_4 = np.power(rd_2, 2)
            rd_6 = np.power(rd_2, 3)
            rd_8 = np.power(rd_2, 4)
            x = (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8) * x_d
            y = (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8) * y_d

            if int(y) == int(-y0) and x >= -x0 and x <= x0:
                if x < min_x:
                    min_x = x
                    cut_r = -x_d

    start = int(x0 - cut_r)
    end = int(x0 + cut_r) + 1
    for i in range(start, end):
        for j in range(start, end):
            x_d = j - x0
            y_d = i - y0
            rd_2 = np.power(x_d, 2) + np.power(y_d, 2)
            rd_4 = np.power(rd_2, 2)
            rd_6 = np.power(rd_2, 3)
            rd_8 = np.power(rd_2, 4)
            x = (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8) * x_d
            y = (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8) * y_d

            u = int(x + x0)
            v = int(y + y0)
            if (u >= 0 and u < WIDTH) and (v >= 0 and v < HEIGHT):
                dstImg[i, j, 0] = srcImg[v, u, 0]
                dstImg[i, j, 1] = srcImg[v, u, 1]
                dstImg[i, j, 2] = srcImg[v, u, 2]

                up, down, left, right = collect_edges(start, end, j, i, u, v, up, down, left, right)
                cut_r_gt = int(x0 - up[0][0])
                parameter_c = float(cut_r_gt) / float(cut_r)
                parameter_b = float(final_lon) / float(cut_r_gt * 2)
                det_uv[v, u, 0] = (((parameter_c * x_d) + x0) - u) * parameter_b
                det_uv[v, u, 1] = (((parameter_c * y_d) + y0) - v) * parameter_b

    cropImg = dstImg[int(x0) - int(cut_r):int(x0) + int(cut_r), int(y0) - int(cut_r):int(y0) + int(cut_r)]
    dstImg2 = cv.resize(cropImg, (final_lon, final_lon), interpolation=cv.INTER_LINEAR)

    det_u = det_uv[:, :, 0]
    det_v = det_uv[:, :, 1]

    det_u = compensate(det_u)
    det_v = compensate(det_v)

    source_Img, det_u, det_v = get_new_gt2(srcImg, det_u, det_v, up, down, left, right)
    source_Img = source_Img[int(x0) - int(cut_r_gt):int(x0) + int(cut_r_gt),
                 int(y0) - int(cut_r_gt):int(y0) + int(cut_r_gt)]
    source_Img = cv.resize(source_Img, (final_lon, final_lon), interpolation=cv.INTER_LINEAR)

    return dstImg, dstImg2, source_Img


path = 'H:/PCN-main/data_prepare/picture/'
k1_top = 1e-4
k1_down = 1e-6

k2_top = 1e-9
k2_down = 1e-11

k3_top = 1e-14
k3_down = 1e-16

k4_top = 1e-19
k4_down = 1e-21
num = 1

if __name__ == "__main__":
    for root, dirs, img_list in os.walk(path):
        print(img_list)
    for files in img_list:
        k1 = random.uniform(k1_down, k1_top)
        k2 = random.uniform(k2_down, k2_top)
        k3 = random.uniform(k3_down, k3_top)
        k4 = random.uniform(k4_down, k4_top)
        print(files)
        file_path = os.path.join('H:/PCN-main/data_prepare/picture/', files)
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue

        srcImg = cv.imread(file_path)
        if srcImg is None:
            print(f"无法读取文件: {file_path}")
            continue

        srcImg = cv.resize(srcImg, (400, 400))

        # 定义级联模型和权重，使用 functools.partial 预设参数
        models = [lambda x: optical_distortion(x, k1, k2, k3, k4),
                  partial(geometric_distortion, factor=0.2),
                  operation_distortion,
                  overlay_distortion]
        betas = [0.25, 0.25, 0.25, 0.25]

        # 应用级联模型融合
        rd = np.random.rand() * 2 + 1  # 随机输入数据在 [1, 3] 范围内
        ru = cascaded_model_fusion(rd, models, betas)
        print("畸变后的输出数据: ", ru)

        dstImg, cutImg, source_Img = create_fish(srcImg, k1, k2, k3, k4)

        cv.imwrite(f'../dataset/data/train/{num}.jpg', cutImg)
        cv.imwrite(f'../dataset/gt/train/{num}.jpg', source_Img)
        num = num + 1
