import numpy as np
from numpy.polynomial import polynomial as poly

# 参数
n = 256
q = 8380417

# 在 CRYSTALS-
# Dilithium 数字签名方案中，模 q 和 n 的值是固定的，
# 其中，n = 256，q = 223 - 213 + 1。


d = 13
tau = 60
eta = 2


def ntt(a):
    # 数论变换（简化）
    # 计算实输入的一维离散傅里叶变换。
    return np.fft.rfft(a)


def intt(a):
    # 逆数论变换（简化）
    # 计算rfft的逆。
    return np.fft.irfft(a, n=n).real.astype(np.int32)


def mod_q(x):
    # 对输入 x 进行模 q 运算
    return np.mod(x.astype(np.int64), q).astype(np.int32)


'''np.mod 函数
功能： np.mod(a, b) 返回数组 a 中每个元素与标量 b 进行模运算的结果。
返回值:返回与输入 a 相同形状的数组，其中每个元素都被计算为 a[i] % b'''

'''.astype() 方法是 NumPy 数组中的一个功能，用于将数组的数据类型转换为指定的类型。以下是关于 .astype() 方法的详细解释：
功能:
数据类型转换： .astype(dtype) 将数组中的元素转换为指定的数据类型 dtype。
参数:
dtype:要转换成的目标数据类型，可以是 NumPy 的数据类型（如 np.int32、np.float64、np.str 等），也可以是 Python 的基本类型（如 int、float、str）。
返回值:
返回一个新的数组，具有相同的数据，但数据类型已被转换为指定的类型。
例:
import numpy as np

# 创建一个浮点数数组
arr = np.array([1.5, 2.3, 3.7])

# 将数组转换为整数类型                               np.int32:32 位整数，范围为 -2147483648 到 2147483647
int_arr = arr.astype(np.int32)   使用 .astype(np.int32) 将其转换为 32 位整数类型，得到的新数组 int_arr 中的元素都是整数。

print(int_arr)  # 输出: [1 2 3]    .astype() 返回的是一个新数组，原数组保持不变。'''


def poly_mul_mod(a, b):
    # 对两个多项式 a 和 b 进行乘法运算，然后对结果进行填充、折叠和模约简
    # 多项式乘法
    c = poly.polymul(a, b)
    # 填充多项式结果
    c_padded = np.pad(c, (0, 2 * n - len(c)))  # 在 c 的末尾填充零，使得填充后的长度达到 2 * n
    # 折叠并减去（模拟多项式环上的乘法）
    return mod_q(c_padded[:n] - c_padded[n:2 * n])
    # c_padded[:n] - c_padded[n:2 * n]：对应元素相减，这通常用于在多项式环 Z_q[x]/(x^n - 1) 中模拟乘法和约简。
    # mod_q(c_padded[:n] - c_padded[n:2 * n])中的每一个元素都是在环 Rq = ℤq[ x ] /( Xn + 1 )上的多项式

    # 对应step 1:，A 中的每一个元素都是在环
    # Rq =ℤq[ x ] /( Xn + 1 )上的多项式，其中 q = 223-213 + 1，n = 256


def generate_keypair():
    # 生成了一对多项式系数 s 和 a，并通过多项式乘法、添加噪声和模约简等操作计算了一系列相关的值

    # 生成一个密钥对，包括秘密密钥s（一个小的多项式）和公钥a（一个随机的多项式），以及一个与秘密密钥相关的多项式t（通常是a和s的乘积加上一个小的误差项）。
    # 公钥用于加密和验证签名，秘密密钥用于解密和生成签名。

    s = np.random.randint(-eta, eta + 1, size=n, dtype=np.int32)  # 一个长度为 n 的多项式系数数组，其元素随机选取在 [-eta, eta] 范围内的整数
    a = np.random.randint(0, q, size=n, dtype=np.int32)  # 一个长度为 n 的多项式系数数组，其元素随机选取在 [0, q-1] 范围内的整数
    ''' np.random.randint 是 NumPy 库中的一个函数，用于生成随机整数.
        numpy.random.randint(low, high=None, size=None, dtype=int)
        low: 生成随机数的下界（包含该值）。
        high: 生成随机数的上界（不包含该值）。如果只提供 low,则将其视为上界，生成 [0, low) 范围内的随机整数。
        size: 输出数组的形状。如果是单个整数，则生成一个一维数组；如果是元组，例如 (m, n)，则生成一个 m 行 n 列的二维数组。
        dtype: 输出数组的数据类型，默认为 int.'''

    # 对应step 2：利用 SHAKE-256 算法和种子分别产生l 维向量 s1 和 k 维向量 s2，其中向量 s1 和 s2 中的每一个元素都是-η 到 η 中的随机数。

    # 调用 poly_mul_mod 函数计算多项式 a 和 s 的乘积，并对结果进行模 q 约简，得到 p_m_m
    # p_m_m = poly_mul_mod(a, s)
    # 在 p_m_m 的基础上添加一个随机噪声，噪声的每个元素也是随机选取在 [-eta, eta] 范围内的整数。
    '''poly_mul_mod(a, s):这是一个假设存在的函数，用于执行多项式乘法。它会对数组 a 和 s 进行多项式乘法计算。结果是一个多项式的系数数组，通常是一个新的数组。'''
    # p_m_m_1 = p_m_m + np.random.randint(-eta, eta + 1, size=n, dtype=np.int32)
    # 对添加噪声后的结果 p_m_m_1 进行模 q 约简，得到 m_q
    # m_q = mod_q(p_m_m_1)
    # t = m_q

    t = mod_q(poly_mul_mod(a, s) + np.random.randint(-eta, eta + 1, size=n, dtype=np.int32))

    '''mod_q(...):这是另一个假设存在的函数，通常用于执行模运算。它将输入值进行模 q 运算，确保结果在某个特定的范围内，通常是 [0, q-1]。
    这一步是为了将结果限制在特定的数值范围，常见于密码学或编码理论中。'''
    # 对应Step 3:计算向量 t = As1 + s2

    return (s, a), t
    # return s, a, p_m_m, p_m_m_1, m_q


def generate_key_shares(t, num_shares, threshold):
    # 一个多项式 t（其常数项是要共享的秘密），生成 num_shares 个密钥份额，每个份额都与一个唯一的标识符 i 相关联。
    # 这些份额是根据门限秘密共享方案生成的，其中 threshold 是重构秘密所需的最小份额数。
    # 只有当收集到至少 threshold 个份额时，才能使用拉格朗日插值法或其他方法重构出原始秘密。

    coeffs = np.random.randint(0, q, size=(threshold - 1, n), dtype=np.int32)
    coeffs = np.vstack([t, coeffs])

    # 初始化系数矩阵
    # 首先，生成一个大小为 (threshold - 1) x n 的随机整数矩阵 coeffs，其元素在 [0, q-1] 范围内。
    # 然后，将多项式 t（一个长度为 n 的数组）作为第一行添加到 coeffs 矩阵的顶部，形成一个新的矩阵。现在，coeffs 的第一行是 t，其余行是随机生成的系数。
    '''np.vstack:这个函数用于垂直堆叠数组。它将多个数组沿着垂直方向（行的方向）组合在一起。结果赋值：将堆叠后的结果赋值回 coeffs 变量。
    例:
    import numpy as np
    t = 3
    coeffs = np.array([1, 2, 3])
    coeffs = np.vstack([t, coeffs])
    print(coeffs)

    输出:
    [[3]
     [1]
     [2]
     [3]]
    这代表 t 成为了新数组的第一行，而原来的 coeffs 成为了后面的行.'''

    # 对应Step 1：生 成 系 数 小 于 γ1 的 多 项 式 y 的 屏 蔽 向量(masking vector)，参 数 γ1 需 要 设 置 在 一 定 范 围 内
    # 使得最终签名不会泄露密钥(即签名算法是零知识的)，且使得签名不容易被伪造。

    shares = []
    # 生成密钥份额
    for i in range(1, num_shares + 1):
        share = np.zeros(n, dtype=np.int32)
        for j in range(threshold):  # 这里有一个嵌套循环，它从 0 到 threshold - 1 进行迭代
            share = mod_q(share + coeffs[j] * (i ** j))  # coeffs[j] * (i ** j)：将当前的系数 coeffs[j] 和当前索引 i 的 j 次方相乘。
        shares.append((i, share))
    # 初始化一个空列表 shares 来存储生成的密钥份额。
    # 对于每个 i 从 1 到 num_shares（包括 num_shares），执行以下操作：
    # 初始化一个长度为 n 的零数组 share。
    # 对于每个 j 从 0 到 threshold - 1，计算 coeffs[j] * (i ** j)，并将结果加到 share 上。
    # 注意，这里 coeffs[j] 是一个长度为 n 的数组，所以这个操作是逐元素进行的。
    # 对 share 应用 mod_q 函数进行模约简，确保其结果在 [0, q-1] 范围内。
    # 将 (i, share) 添加到 shares 列表中。这里 i 是份额的标识符（通常是参与者的标识或索引），share 是对应的密钥份额。
    return shares


def sign_partial(s, message, a):
    # 一个基于多项式运算和随机数生成的部分签名过程
    # s: 一个多项式，通常表示私钥或某个秘密值。
    # message: 消息，虽然在此代码段中并未直接使用，但在实际应用中通常会参与计算或以某种方式影响签名过程。
    # a: 一个多项式，通常与公钥或公开参数相关。
    y = np.random.randint(-2 ** d, 2 ** d, size=n, dtype=np.int32)
    # 生成一个大小为n的整数数组y，其中每个元素随机选取在-2^d到2^d - 1之间（包含边界）
    # d，n定义了多项式的维度和系数的范围
    w = mod_q(ntt(poly_mul_mod(a, y)))
    # 计算多项式a和y的乘积，应用数论变换，对结果进行模q运算，结果存储在w中，它将是后续计算的一个中间值
    c = np.random.choice([-1, 0, 1], size=n, p=[1 / (2 * tau), 1 - 1 / tau, 1 / (2 * tau)])
    # 生成一个大小为n的数组c，其中每个元素是-1、0或1，按照给定的概率分布选择：-1和1的概率各为1 / (2 * tau)，0的概率为1 - 1 / tau
    z = mod_q(y + poly_mul_mod(s, c))
    # 结果z是部分签名的一个重要组成部分
    return z, c


def combine_signatures(partial_sigs, indices, t):
    # partial_sigs: 一个包含部分签名的列表，每个部分签名是一个多项式表示的数组。
    # indices: 一个与partial_sigs对应的索引列表，用于拉格朗日插值。
    # t: 一个目标值，通常表示要计算插值的点（在此上下文中可能是某个特定的值，如0，用于签名聚合）。
    lambda_i = lagrange_interpolation(indices, 0, q)
    # 调用lagrange_interpolation函数计算拉格朗日插值系数lambda_i。
    z = np.zeros(n, dtype=np.int32)
    # 初始化一个全零数组z作为最终的签名结果。
    for (i, partial_z), l in zip(partial_sigs, lambda_i):
        z = mod_q(z + l * partial_z)
    # 遍历partial_sigs和lambda_i，对每个部分签名和对应的插值系数，计算l * partial_z，并将结果累加到z上，同时对结果进行模q运算
    return z
    # 返回一个多项式表示的数组z，作为完整的签名


def verify_signature(t, message, z, c, a):
    # t: 一个值，通常与消息或签名方案中的某个参数相关。
    # message: 消息本身，虽然在此函数实现中并未直接使用，但在实际应用中通常会参与验证过程。
    # z: 签名，即combine_signatures函数的输出。
    # c: 一个多项式，通常与签名生成过程中的随机数相关。
    # a: 一个多项式，通常与公钥或公开参数相关。
    w1 = mod_q(ntt(poly_mul_mod(a, z)))
    # 签名z与多项式a的乘积的NTT（数论变换）结果，并对结果进行模q运算
    w2 = mod_q(ntt(poly_mul_mod(t, c)))
    # t与c的乘积的NTT结果，并对结果进行模q运算
    return np.allclose(mod_q(w1 - w2), mod_q(ntt(poly_mul_mod(a, z) - poly_mul_mod(t, c))))
    # 比较mod_q(w1 - w2)与mod_q(ntt(poly_mul_mod(a, z) - poly_mul_mod(t, c)))是否接近（使用np.allclose函数），以验证签名是否正确


def lagrange_interpolation(indices, x, prime):
    # 用于计算拉格朗日插值系数
    # indices: 一个索引列表，用于指示插值点的位置。
    # x: 一个值，表示要计算插值的点。
    # prime: 一个大素数，用于模运算
    result = []
    for i in indices:
        # 对于每个索引i，计算插值系数。这涉及遍历所有索引j（j != i），并计算分子和分母
        numerator, denominator = 1, 1
        for j in indices:
            if i != j:
                # 分子是(x - j)的连乘积，分母是(i - j)的连乘积，然后对分母取模prime的逆元
                numerator = (numerator * (x - j)) % prime
                denominator = (denominator * (i - j)) % prime
        # 最后，将分子与分母的逆元相乘，并对结果取模prime
        # 返回一个系数列表，用于拉格朗日插值
        result.append((numerator * pow(denominator, -1, prime)) % prime)
    return result


# Example usage
threshold = 3
num_shares = 5

# Key generation
(s, a), t = generate_keypair()
shares = generate_key_shares(t, num_shares, threshold)

# Signing
message = "Hello, threshold signature!"
partial_sigs = []
for i in range(threshold):
    z, c = sign_partial(shares[i][1], message, a)
    partial_sigs.append((shares[i][0], z))

# Combining signatures
indices = [share[0] for share in shares[:threshold]]
combined_z = combine_signatures(partial_sigs, indices, t)

# Verification
is_valid = verify_signature(t, message, combined_z, c, a)

print("Is the signature valid?", is_valid)
