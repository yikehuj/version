from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import numpy as np
import random

# 定义参数
q = 8380417
n = 256
k, l = 3, 5  # 你可以根据需要设置k和l的值
seed = b'your-seed-here'  # 种子，必须是字节类型
seed_s1 = b'my-s1'
seed_s2 = b'my-s2'


def mod_q(x):
    # 对输入 x 进行模 q 运算
    # 只保留实数部分
    return np.mod(x.real.astype(np.int64), q % 400099).astype(np.int32)


# 使用SHAKE-256生成随机比特流
def shake256_random_bytes(seed_input, length):
    digest = hashes.Hash(hashes.SHAKE256(length))
    digest.update(seed_input)
    return digest.finalize()


# 生成一个环R_q上的随机多项式
def generate_polynomial(seed, q, n):
    # 为了简单起见，我们生成一个长度为n的整数数组，每个整数在0到q-1之间
    # 实际上，你可能需要更复杂的逻辑来确保多项式满足特定的性质（如不可约性）
    # 但在这个例子中，我们只需要随机性
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

# 打印矩阵A（只打印前几行和前几个多项式以避免输出过长）
for i in range(k):
    for j in range(l):
        print(f"A[{i}][{j}] = {A[i][j]}")
        print(len(A[i][j]))
# print('\n')
# print(A)