import numpy as np


def ntt(a, n):
    # 数论变换（简化）
    # 计算实输入的一维离散傅里叶变换。
    return np.fft.rfft(a, n=n).real.astype(np.int32)


def intt(a, n):
    # 逆数论变换（简化）
    # 计算rfft的逆。
    return np.fft.irfft(a, n=n).real.astype(np.int32)


def generate_masking_vector(y, l, n):
    masking_vector = []
    for i in range(l):
        y1 = np.random.randint(-2 ** y, 2 ** y, size=n, dtype=np.int32)
        masking_vector.append(y1)
    return np.array(masking_vector)


# l, n = 4, 256
# y = 17
# masking_vector = generate_masking_vector(y, l, n)
# print(masking_vector)
# print('\n')
# print(len(masking_vector[0]))

def operation_NTT(a, y, n):  # 使用有问题。
    return intt(ntt(a, n) * ntt(y, n), n)


def Polynomial_A_Y(a, y, k, l):
    w = []  # 储存w
    result = []
    for i in range(k):
        for j in range(l):
            result.append(np.polymul(a[i][j], y[j]))
        temp = 0
        for col in range(l):
            temp = temp + result[col + i * l]
        w.append(temp)
    return np.array(w)
