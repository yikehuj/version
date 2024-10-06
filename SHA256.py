from array import array

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import numpy as np
import random

# 定义参数
q = 8380417
n = 256
k, l = 4, 4  # 你可以根据需要设置k和l的值
seed = b'your-seed-here'  # 种子，必须是字节类型
seed_s1 = b'my-s1'
seed_s2 = b'my-s2'
eta = 2

def mod_q(x):
    # 对输入 x 进行模 q 运算
    # 只保留实数部分
    return np.mod(x.real.astype(np.int64), q).astype(np.int32)


# 使用SHAKE-256生成随机比特流
def shake256_random_bytes(seed_input, length):
    digest = hashes.Hash(hashes.SHAKE256(length))
    digest.update(seed_input)
    return digest.finalize()


# 生成一个环R_q上的随机多项式
def generate_polynomial(seed, q, n):
    length = n * 4  # 每个系数用4个字节表示（足够容纳q的范围）
    random_bytes = shake256_random_bytes(seed, length)
    coefficients = [int.from_bytes(random_bytes[i:i + 4], 'big') % q for i in range(0, length, 4)]
    # 截断到n个系数（如果生成了多余的系数）
    return coefficients[:n]


# 生成k x l矩阵A，其中每个元素是环R_q上的多项式
def generate_matrix_A(k, l, seed, q, n):
    matrix = []
    for i in range(k):
        row = []
        for j in range(l):
            # 为每个多项式生成一个新的种子（基于原始种子和索引）
            polynomial_seed = seed + (i * l + j).to_bytes(4, 'big')  # 简单地使用索引来生成不同的种子
            polynomial = generate_polynomial(polynomial_seed, q, n)
            row.append(polynomial)
        matrix.append(row)
    return np.array(matrix)


# 生成矩阵A
A = generate_matrix_A(k, l, seed, q, n)

# 生成s1 s2


def generate_random_vector(seed, dimension, eta, n):
    length = n * 4
    random_bytes = shake256_random_bytes(seed, length)
    max_value = 256  # 一个字节的最大值（2^8）
    scale_factor = eta * 2 / (max_value - 1)
    offset = -eta

    # 生成随机向量
    random_vector = [(b // 1) * scale_factor + offset for b in random_bytes[:dimension]]
    random_vector = [(b % 5 - 2) for b in random_bytes[:dimension]]

    return np.array(random_vector)


def function_As(a, s, k, l):
    # 多项式矩阵与向量相乘：向量元素为常数，k * l矩阵中每个元素内的多项式的每一项乘以该常数
    # t[n] = a[n][j] * s[j] (j from 0 to l)

    t = []                     # 接收计算好的Asi
    array_as = []
    time = 0
    for row in range(k):
        for col in range(l):
            t.append(a[row][col] * s[col])
        temp = 0
        for i in range(l):
            temp = temp + t[i + row * l]
            # print('___________________________________________________________\n',i + 1, s[i])
            # print(temp)
            # print('\n')

        array_as.append(temp)
    return np.array(array_as)
    # 乘积为k * 1的向量

def add_t_s(a_s, s, k, n):
    for i in range(k):
        a_s[i][n - 1] = a_s[i][n - 1] + s[i]
    return a_s
