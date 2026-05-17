# CSC3060 Project 5 Bonus 最新评分要求整理

## 1. Bonus 总分与分档

Bonus 最高 **30 分**。

评分仍然按照 geometric mean speedup，也就是 `GM` 分档：

| 最新 GM 区间 | Bonus 分数 |
|---:|---:|
| `1.15 ≤ GM < 1.25` | 5 |
| `1.25 ≤ GM < 1.40` | 10 |
| `1.40 ≤ GM < 1.60` | 15 |
| `1.60 ≤ GM < 2.00` | 20 |
| `GM ≥ 2.00` | 30 |

其中，**满分要求是：**

```text
GM >= 2.00
```

但这里的 `GM` 已经不是原 PDF 中的：

```text
T_baseline / T_bonus
```

而是最新规则下的 **lower-bound-normalized GM**。

---

## 2. 最新 bonus multiplier 公式

对每个 kernel，定义：

| 符号 | 含义 |
|---|---|
| `T_naive,i` | 第 `i` 个 kernel 的 naive runtime |
| `T_bonus,i` | 第 `i` 个 kernel 的 bonus optimized runtime |
| `LB_i` | 第 `i` 个 kernel 的 `NAIVE_SPEEDUP_LOWER_BOUND_*` |
| `m_i` | 第 `i` 个 kernel 对 bonus GM 的 multiplier |

最新每个 kernel 的 multiplier 是：

```text
m_i = T_naive,i / (T_bonus,i * LB_i)
```

也可以写成：

```text
m_i = (T_naive,i / T_bonus,i) / LB_i
```

也就是说：

```text
先算 raw speedup over naive:
    raw_i = T_naive,i / T_bonus,i

再除以该 kernel 的 lower bound:
    m_i = raw_i / LB_i
```

---

## 3. 最新 bonus GM 公式

如果有 `n` 个有效 kernel 参与 bonus GM，则：

```text
GM = (m_1 * m_2 * ... * m_n)^(1/n)
```

代入 multiplier 公式：

```text
GM = Π_i [T_naive,i / (T_bonus,i * LB_i)] 的 n 次方根
```

更完整地写：

```text
GM = (
    ∏_{i=1}^{n} T_naive,i / (T_bonus,i * LB_i)
)^(1/n)
```

---

## 4. 满分 30 分的实际含义

满分要求：

```text
GM >= 2.00
```

也就是：

```text
geometric_mean(T_naive / (T_bonus * LB)) >= 2.00
```

直观理解：

> Bonus 满分要求你的 bonus 版本在几何平均意义上，比 basic lower-bound 目标再快 **2 倍**。

因为 basic 目标是：

```text
T_naive / T_stu > LB
```

而 bonus 满分大致对应：

```text
T_naive / T_bonus ≈ 2 * LB
```

注意：这是**几何平均意义上**的要求，不是每个 kernel 都必须单独达到 `2 * LB`。

---

## 5. 各 kernel 不拖满分 GM 的参考 raw speedup

如果希望某个 kernel 对满分 GM 不拖后腿，它的 raw speedup over naive 最好达到：

```text
T_naive / T_bonus >= 2 * LB
```

| Kernel | LB | 对满分 GM 不拖后腿的 raw speedup 参考值 |
|---|---:|---:|
| bitwise | 8.00 | `16.00x` |
| blackscholes | 1.35 | `2.70x` |
| filter_gradient | 1.45 | `2.90x` |
| graph | 2.50 | `5.00x` |
| grff | 3.15 | `6.30x` |
| image_proc | 1.73 | `3.46x` |
| matmul | 2.45 | `4.90x` |
| relu | 2.50 | `5.00x` |
| sparse_spmm | 1.40 | `2.80x` |
| trace_replay | 1.75 | `3.50x` |

例如 ReLU：

```text
LB = 2.50

如果:
    T_naive / T_bonus = 5.00

那么:
    multiplier = 5.00 / 2.50 = 2.00

这个 kernel 对满分 GM 不拖后腿。
```

---

## 6. GM 是几何平均，不是每个 kernel 单独评分

Bonus 不是逐 kernel 单独给分，而是看整体 GM。

例如三个 kernel 的 multiplier 分别是：

```text
m1 = 1.5
m2 = 2.0
m3 = 2.8
```

那么：

```text
GM = (1.5 * 2.0 * 2.8)^(1/3)
   ≈ 2.03
```

虽然 `m1 < 2.00`，但整体仍然可以达到满分区间。

但是几何平均对低值很敏感。如果某个 kernel multiplier 很低，例如 `0.5` 或 `0.8`，它会明显拉低整体 GM。

---

## 7. Correctness 要求

Bonus 的第一前提仍然是：

```text
correctness passed
```

如果某个 bonus kernel 没有通过 checker，那么该 kernel 的性能结果无效。

实际顺序应该理解为：

```text
1. 先 correctness check。
2. correctness passed 后，才计算 T_naive / (T_bonus * LB)。
3. 所有有效 kernel 的 multiplier 进入 GM。
```

---

## 8. Bonus 允许的优化方式

Bonus 部分允许比 basic 更激进的优化，包括：

```text
1. 自定义 CMakeLists.txt
2. 使用额外编译参数
3. 使用外部库
4. 使用 OpenMP / std::thread 多线程
5. 使用 SIMD intrinsics 或 compiler vectorization hints
6. 使用 GPU / CUDA
7. 使用更高级 profiling 工具，例如 VTune / Nsight
```

但必须满足：

```text
1. 能在 course server 上自动编译成功。
2. 不需要 grader 手动安装或配置额外环境。
3. 所有非标准编译流程、依赖、flags 都要在 report 中说明。
4. 代码必须保持 correctness。
```

---

## 9. Bonus 提交结构

如果尝试 bonus，bonus 相关文件应放入单独目录：

```text
bonus/
    CMakeLists.txt
    <bonus kernel>.h
    <bonus kernel>.cpp
    ...
```

完整提交结构建议：

```text
SID1_SID2_CodeOpt/
    CMakeLists.txt
    include/
    src/
    report.pdf
    ChatWithAI.pdf
    bonus/
        CMakeLists.txt
        <bonus kernel>.h
        <bonus kernel>.cpp
        ...
```

---

## 10. Bonus 报告中建议放的表格

你的 bonus report 最好按下面格式写：

| Kernel | `T_naive` | `T_bonus` | LB | Raw speedup `T_naive/T_bonus` | Multiplier `Raw/LB` |
|---|---:|---:|---:|---:|---:|
| bitwise | ... | ... | 8.00 | ... | ... |
| blackscholes | ... | ... | 1.35 | ... | ... |
| filter_gradient | ... | ... | 1.45 | ... | ... |
| graph | ... | ... | 2.50 | ... | ... |
| grff | ... | ... | 3.15 | ... | ... |
| image_proc | ... | ... | 1.73 | ... | ... |
| matmul | ... | ... | 2.45 | ... | ... |
| relu | ... | ... | 2.50 | ... | ... |
| sparse_spmm | ... | ... | 1.40 | ... | ... |
| trace_replay | ... | ... | 1.75 | ... | ... |

然后写：

```text
Geometric mean of normalized multipliers = ...
Bonus points = ...
```

---

## 11. 计算流程模板

对每个 kernel：

```text
raw_i = T_naive,i / T_bonus,i
m_i   = raw_i / LB_i
```

对所有有效 kernel：

```text
GM = geometric_mean(m_i)
```

根据 GM 查分档：

```text
1.15 <= GM < 1.25  -> 5 bonus points
1.25 <= GM < 1.40  -> 10 bonus points
1.40 <= GM < 1.60  -> 15 bonus points
1.60 <= GM < 2.00  -> 20 bonus points
GM >= 2.00         -> 30 bonus points
```

---

## 12. 最终总结

最新 bonus 评分可以压缩成一句话：

```text
Bonus 评分看 lower-bound-normalized GM:
    GM = geometric_mean(T_naive / (T_bonus * LB))

满分 30 分要求:
    GM >= 2.00
```

也就是说，不能再用旧公式：

```text
T_baseline / T_bonus
```

来估计 bonus 分数；现在必须用：

```text
T_naive / (T_bonus * LB)
```

来计算每个 kernel 的 multiplier，再对所有有效 kernel 做 geometric mean。
