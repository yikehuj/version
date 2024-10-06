from array import array

import numpy as np
from numpy.polynomial import polynomial as poly
from SHA256 import generate_matrix_A
from SHA256 import generate_random_vector, function_As, add_t_s

# 参数
n = 256
q = 8380417

# 在 CRYSTALS-
# Dilithium 数字签名方案中，模 q 和 n 的值是固定的，
# 其中，n = 256，q = 223 - 213 + 1。


d = 13
tau = 60
eta = 2


seed_s1 = b'my-s1'
seed_s2 = b'my-s2'

k, l = 4, 4  # 你可以根据需要设置k和l的值
seed = b'your-seed-here'  # 种子，必须是字节类型

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
    # 只保留实数部分
    return np.mod(x.real.astype(np.int64), q).astype(np.int32)


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


def generate_keypair():
    # 生成公钥 pk = (A, t)，私钥 sk = (A, t, s1, s2)。
    a = np.random.randint(0, q, size=n, dtype=np.int32)
    # 对应step 1:，A 中的每一个元素都是在环
    # Rq =ℤq[ x ] /( Xn + 1 )上的多项式，其中 q = 223-213 + 1，n = 256
    s1 = np.random.randint(-eta, eta + 1, size=n, dtype=np.int32)
    # 一个长度为 n 的多项式系数数组，其元素随机选取在 [-eta, eta] 范围内的整数
    s2 = np.random.randint(-eta, eta + 1, size=n, dtype=np.int32)
    # 对应step 2：利用 SHAKE-256 算法和种子分别产生l 维向量 s1 和 k 维向量 s2，其中向量 s1 和 s2 中的每一个元素都是-η 到 η 中的随机数。
    t = mod_q(poly_mul_mod(a, s1) + s2)
    # 对应Step 3:计算向量 t = As1 + s2
    return (s1, a), t
    # 公钥 pk = (A, t)，私钥 sk = (A, t, s1, s2)。

    # 问题1：
    # A在生成上没有按照：利用 SHAKE-256 算法和种子产生
    # k × l 维的矩阵 A，A 中的每一个元素都是在环 Rq =ℤq[ x ] /( Xn + 1 )上的多项式，
    # 其中 q = 223-213 + 1， n = 256。
    # 问题2：
    # s1, s2没有利用 SHAKE-256 算法和种子分别产生
    # l 维向量 s1 和 k 维向量 s2，其中向量 s1 和 s2 中的每一 个元素都是-η 到 η 中的随机数。


def generate_key_shares(t, num_shares, threshold):
    # 使用Shamir的秘密共享方案，将一个多项式形式的秘密t分割成多个份额，并返回这些份额的列表。
    # 每个份额都是通过对一个多项式在某个点上的值进行计算并取模得到的。
    # 只有当收集到足够的份额（至少threshold个）时，才能通过拉格朗日插值等方法重构出原始的秘密多项式。

    # t：这是一个整数数组，表示要共享的秘密（它是一个多项式形式的秘密）。
    # num_shares：整数，表示要生成的秘密份额的总数。
    # threshold：整数，表示重构秘密所需的最小份额数（也即门限值）。

    coeffs = np.random.randint(0, q, size=(threshold - 1, n), dtype=np.int32)
    coeffs = np.vstack([t, coeffs])

    # 生成一个(threshold-1) x n的随机整数矩阵coeffs，其中每个元素都在[0, q-1]范围内。
    # 这里q是一个预定义的大素数，用于模运算以保证数值不会太大。n是多项式的阶数，也即秘密和份额多项式的最高次项的次数加一
    # 使用np.vstack([t, coeffs])将秘密t和随机生成的系数矩阵coeffs垂直堆叠，形成一个threshold x n的矩阵。
    # 这里的coeffs实际上是多项式的系数，其中t是常数项（秘密的多项式表示中的最低次项系数）
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

    shares = []
    # 为每个参与者生成一个秘密份额
    for i in range(1, num_shares + 1):
        # 外层循环遍历从1到num_shares，为每个参与者生成一个唯一的标识符i（这里i实际上是参与者的ID或序号，从1开始）
        share = np.zeros(n, dtype=np.int32)
        for j in range(threshold):
            # 内层循环计算份额
            share = mod_q(share + coeffs[j] * (i ** j))
            # 对于每个i，计算多项式f(i)的值，其中f(x)是以coeffs为系数的多项式。
            # 计算f(i)的过程是通过累加coeffs[j] * (i ** j)完成的，这里j从0到threshold-1，对应多项式的各项系数。
            # 每次累加后都通过mod_q函数进行模q运算，以保持数值在[0, q-1]范围内
        shares.append((i, share))
        # 每个份额是一个元组(i, share)，其中i是参与者的标识符，share是计算得到的秘密份额（一个长度为n的整数数组）
    return shares


def sign_partial(s, message, a):
    # 阈值签名方案中的一个关键部分，它负责生成部分签名。通过结合多个部分签名，可以生成最终的签名，从而实现对消息的验证。

    # 一个基于多项式运算和随机数生成的部分签名过程
    # s: 一个多项式，通常表示私钥或某个秘密值。
    # message: 消息，虽然在此代码段中并未直接使用，但在实际应用中通常会参与计算或以某种方式影响签名过程。
    # a: 一个多项式，通常与公钥或公开参数相关。
    y = np.random.randint(-2 ** d, 2 ** d, size=n, dtype=np.int32)
    # 这行代码生成一个长度为 n 的随机整数数组 y，其元素取值范围在 -2**d 到 2**d 之间（不包括 2**d）。
    # 这个随机数组在后续的签名生成中起到掩蔽的作用。
    # Step 1 生成系数小于γ1的多项式y的屏蔽向量(masking vector)，参数γ1需要设置在一定范围内
    # 使得最终签名不会泄露密钥(即签名算法是零知识的)，且使得签名不容易被伪造。
    w = mod_q(ntt(poly_mul_mod(a, y)))
    # 计算多项式a和y的乘积，应用数论变换，对结果进行模q运算，结果存储在w中，它将是后续计算的一个中间值
    # Step 2 计算 Ay，并使用 Decompose q ( · )算法得到 w 的高位比特 w1 和低位比特 w2，分解时使用的α = 2γ2
    c = np.random.choice([-1, 0, 1], size=n, p=[1 / (2 * tau), 1 - 1 / tau, 1 / (2 * tau)])
    # 生成一个大小为n的数组c，其中每个元素是-1、0或1，按照给定的概率分布选择：-1和1的概率各为1 / (2 * tau)，0的概率为1 - 1 / tau
    # Step 3 使用哈希函数 H0 计算挑战值 c ∈ C，c 是Rq 中 的 多 项 式 ，系 数 c 0,c 1,⋯,c 255 ∈ {±1,0 }，其 中±1 的个数为 τ。
    # 选择这种分布的原因是 c 具有小的范数，并且来自（拥有足够大的）熵 log 2 ( 256τ ) + τ 的挑战值空间。
    z = mod_q(y + poly_mul_mod(s, c))
    # 这行代码计算部分签名 z。它首先将 y 和 s 与 c 的多项式乘积相加（使用 poly_mul_mod 函数计算 s 和 c 的乘积），然后对结果应用模 q 操作。
    # z 是部分签名的结果，它将与其他部分签名一起用于生成最终的签名。
    # Step 4 计算潜在的签名 z ≔ y + cs1，由于直接输出可能会导致密钥的泄露，因此使用拒绝采样[17]，参数 β 被设置为 cs1 的最大可能系数。
    # 如果z的任何一项系数大于 γ1 - β，那么拒绝并重新开始签名过程 。
    # 同样，如果Az-ct的任何低位比特的系数大于γ2 - β，则需要重新开始计算签名。

    # z为潜在签名   c为挑战值
    return z, c


def lagrange_interpolation(indices, x, prime):
    # 用于计算拉格朗日插值系数
    # indices: 一个包含用于插值的点的索引的列表。
    # x: 要插值的点的x坐标值，通常在Shamir的秘密共享中为0，因为我们想要恢复的是多项式在x=0处的值（即秘密）。
    # prime: 一个质数，用于模运算，以确保结果保持在一定的范围内。
    result = []
    # 初始化一个空列表result，用于存储每个索引对应的拉格朗日系数。
    for i in indices:
        # 对于indices中的每个索引i：
        # 初始化分子numerator和分母denominator为1。
        # 遍历indices中的每个索引j，如果j不等于i：
        # 更新分子：numerator = (numerator * (x - j)) % prime，这里(x - j)是拉格朗日插值公式中的一个因子，% prime确保结果模prime。
        # 更新分母：denominator = (denominator * (i - j)) % prime，这里(i - j)是另一个因子。
        # 计算i对应的拉格朗日系数：(numerator * pow(denominator, -1, prime)) % prime，
        # 其中pow(denominator, -1, prime)是计算denominator在模prime下的逆元。
        # 将计算出的系数添加到result列表中。
        # 对于每个索引i，计算插值系数。这涉及遍历所有索引j（j != i），并计算分子和分母
        numerator, denominator = 1, 1
        for j in indices:
            if i != j:
                numerator = (numerator * (x - j)) % prime
                denominator = (denominator * (i - j)) % prime
        # 返回一个系数列表，用于拉格朗日插值
        result.append((numerator * pow(denominator, -1, prime)) % prime)
    return result


def combine_signatures(partial_sigs, indices, t):
    # 这个函数是基于 Shamir 的秘密共享方案中的拉格朗日插值法来实现的

    # partial_sigs: 一个包含部分签名的列表，每个部分签名是一个多项式表示的数组。
    # indices: 一个与partial_sigs对应的索引列表，用于拉格朗日插值。
    # t: 公钥的一部分，通常是一个多项式，这里用于验证和计算。
    lambda_i = lagrange_interpolation(indices, 0, q)
    # 调用lagrange_interpolation函数计算拉格朗日插值系数lambda_i。
    z = np.zeros(n, dtype=np.int32)
    # 初始化一个全零数组z作为最终的签名结果。
    for (i, partial_z), l in zip(partial_sigs, lambda_i):
        z = mod_q(z + l * partial_z)
    # 同时对结果进行模q运算对于每一个部分签名 partial_z 和对应的系数 l，计算 l * partial_z，然后将结果累加到 z 上。
    # 使用 mod_q 函数确保结果保持在模 q 的范围内。
    return z
    # 返回一个多项式表示的数组z，作为完整的签名


def verify_signature(t, message, z, c, a):
    # 通过比较两个多项式乘积的NTT结果来验证签名的有效性。如果这两个结果相等（或在允许的误差范围内相等），则签名被认为是有效的。

    # t：公钥的一部分，通常是由私钥 s 和一个随机多项式 a 通过多项式乘法生成的。
    # message：要签名的消息。在这个示例代码中，但在实际应用中，消息通常会影响挑战 c 的生成。
    # z：组合后的签名。
    # c：挑战多项式，用于生成签名。
    # a：公钥的另一部分，一个随机多项式。
    w1 = mod_q(ntt(poly_mul_mod(a, z)))
    # 这一步计算 a 和 z 的多项式乘积，然后应用数论变换（NTT）。mod_q 确保结果模 q。这个计算对应于验证方程中的一个部分，即检查 a * z 的NTT结果。
    w2 = mod_q(ntt(poly_mul_mod(t, c)))
    # 这一步计算 t 和 c 的多项式乘积，并应用NTT。这对应于验证方程中的另一个部分，即检查 t * c 的NTT结果。
    return np.allclose(mod_q(w1 - w2), mod_q(ntt(poly_mul_mod(a, z) - poly_mul_mod(t, c))))
    # 比较mod_q(w1 - w2)与mod_q(ntt(poly_mul_mod(a, z) - poly_mul_mod(t, c)))是否接近（使用np.allclose函数），以验证签名是否正确

    # np.allclose 用于检查两个数组是否在给定的容差范围内相等。
    # 在这里，由于我们处理的是整数运算（模 q），理论上应该使用精确相等比较。
    # 但是，由于数值稳定性和实现细节，使用 np.allclose 可能是为了处理潜在的浮点数精度问题（尽管在这个特定场景中，所有运算都是整数运算）。


# Example usage
threshold = 3
num_shares = 5

# Key generation
(s, a), t = generate_keypair()
# 生成一个密钥对，包括秘密密钥s（一个小的多项式）和公钥a（一个随机的多项式），# 以及一个与秘密密钥相关的多项式t（通常是a和s的乘积加上一个小的误差项）。
# 密钥s（一个小的多项式）和公钥a（一个随机的多项式），以及一个与秘密密钥相关的多项式t
shares = generate_key_shares(t, num_shares, threshold)                                   # 输入t，参与人数，门限值，输出秘密份额

# for k in range(num_shares):
#     print(shares[k])
# 查看秘密份额


# Signing
message = "Hello, threshold signature!"                                                  # 要加密的消息
partial_sigs = []                                                                        # 储存秘密份额和签名的数组
for i in range(threshold):                                                               # 循环输出部分签名
    z, c = sign_partial(shares[i][1], message, a)                                        # 输出对应的签名和挑战值
    partial_sigs.append((shares[i][0], z))                                               # 存入数组

# Combining signatures
indices = [share[0] for share in shares[:threshold]]                                     # 创建索引列表，用于拉格朗日插值
combined_z = combine_signatures(partial_sigs, indices, t)                                # 返回完整的签名

# Verification
is_valid = verify_signature(t, message, combined_z, c, a)                                # 通过比较两个多项式乘积的NTT结果来验证签名的有效性。
                                                                                         # 如果这两个结果相等（或在允许的误差范围内相等），则签名被认为是有效的。

print("Is the signature valid?", is_valid)

A = generate_matrix_A(k, l, seed, q, n)
print(A)
print('\n')
s1 = generate_random_vector(seed_s1, l, eta, n)
s2 = generate_random_vector(seed_s2, k, eta, n)
print(s1)
print('\n')
print(s2)
print('\n')
array_As = function_As(A, s1, k, l)
print(array_As)
print('\n')
print(add_t_s(array_As, s2, k, n))
print('\n')