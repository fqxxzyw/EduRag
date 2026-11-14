"""
=================================================================
Milvus前置 —— 余弦相似度计算
=================================================================
用 numpy 手算余弦相似度，理解向量相似度的数学原理。
"""
import numpy as np
"""
计算两个向量的余弦相似度（衡量方向相似性，剔除长度影响）

参数：
    vec_a (np.array): 向量A
    vec_b (np.array): 向量B
返回：
    float: 余弦相似度结果（范围[-1,1]，越接近1方向越一致）
公式：
  - 分子：两个向量对应数字相乘再全部相加（点积：同维度数字乘积之和）
  - 分母：A 所有数平方的和再开根号 × B 所有数平方的和再开根号（即A模长*B模长）
  - 余弦相似度 = 两个向量的点积 ÷ 两个向量模长的乘积

A: [0.5, 0.5]
B: [0.7, 0.7]
C: [0.7, 0.5]
D: [-0.6, -0.5]
"""
def get_dot(vec_a, vec_b):
    """计算2个向量的点积，即：2个向量同维度数字乘积之和"""
    if len(vec_a) != len(vec_b):
        raise ValueError("2个向量必须维度数量相同")

    dot_sum = 0
    # zip(vec_a, vec_b): 把两个向量按位置两两配对
    # 每次遍历出的a和b就是维度相同的两个数字
    for a, b in zip(vec_a, vec_b):
        ab = a * b # 同维度数字乘积
        dot_sum += ab # 累加和

    return dot_sum

def get_norm(vec):
    """计算单个向量的模长：对向量的每个数字求平方、再求和、再开根号"""
    sum_square = 0
    for v in vec:
        vv = v * v # 每个数字的平方
        sum_square += vv # 再求和

    # numpy的sqrt函数完成开根号
    ret = np.sqrt(sum_square)
    return ret

def cosine_similarity(vec_a, vec_b):
    """余弦相似度：2个向量的点积 除以 2个向量模长的乘积"""
    result = get_dot(vec_a, vec_b) / (get_norm(vec_a) * get_norm(vec_b))
    return result

if __name__ == '__main__':
    vec_a = [0.5, 0.5]
    vec_b = [0.7, 0.7]
    vec_c = [0.7, 0.5]
    vec_d = [-0.6, -0.5]

    print("ab:", cosine_similarity(vec_a, vec_b))
    print("ac:", cosine_similarity(vec_a, vec_c))
    print("ad:", cosine_similarity(vec_a, vec_d))

"""
================== 运行输出 ==================
ab: 1.0
ac: 0.9863939238321436
ad: -0.995893206467704
===============================================
"""