# CSC3060 Project 5 最新评分规则适配：代码层级修改报告

> 目标读者：负责修改原始仓库代码、benchmark harness、bonus 统计逻辑和实验报告模板的技术同学。  
> 适配方式：**不执行 `git pull`**，而是在原始仓库基础上手工合并最新评分逻辑。  
> 核心变化：basic 部分从主要看固定 baseline 时间，改为主要看 `T_naive / T_stu` 是否超过每个 kernel 的 lower bound；bonus GM 从 `T_baseline / T_bonus` 改为 `T_naive / (T_bonus * LB)`。

---

## 0. 背景与核心变化

原始 PDF 中，basic 部分主要以固定 baseline 时间作为性能目标：

```text
T_stu <= T_baseline
```

最新邮件和 README 对性能测量方式做了调整。现在每个 kernel 新增一个 lower bound，记作 `LB`，basic 部分主要关注：

```text
speedup over naive = T_naive / T_stu
```

现在 basic 目标变为：

```text
T_naive / T_stu > LB
```

bonus 部分的每个 kernel multiplier 也从原来的：

```text
T_baseline / T_bonus
```

改为：

```text
T_naive / (T_bonus * LB)
```

也可以理解为：

```text
(T_naive / T_bonus) / LB
```


---

## 0.1 本报告的检查结论与重要补充

经过再次对照最新 README / `bench.h` / `run_all.cpp` 的实现逻辑，本文档的核心方向是正确的：  
basic 的主指标确实是 `T_naive / T_stu > LB`，bonus/GM 的归一化 multiplier 确实是 `T_naive / (T_stu * LB)`。

但有三点需要特别注明，避免技术实现时误解：

### 补充 1：basic pass/fail 和 GM 不是同一个概念

对于 **basic part**，每个 kernel 的主要目标仍然是逐项判断：

```text
T_naive / T_stu > LB
```

而 `run_all.cpp` 最后打印的 geometric mean speedup 使用的是 lower-bound-normalized multiplier：

```text
T_naive / (T_stu * LB)
```

这个 GM 更接近 bonus/aggregate 口径，不应把它误解成 basic 的唯一判断标准。  
basic 应以每个 kernel 独立是否超过 lower bound 为主。

### 补充 2：`sparse_spmm` 的 input size 在 README/PDF 和当前 `run_all.cpp` 中存在不一致

README/PDF 中写的是：

```text
sparse_spmm default size: 2048 x 2048
```

但当前最新 `src/main/run_all.cpp` 中实际调用类似：

```cpp
initialize_spmm(sparse_args_ref, 512, 512, -1, {}, seed);
```

也就是说，如果严格复现当前最新 `run_all.cpp`，实际测试 size 是 `512 x 512`；  
如果以 README/PDF 的文字说明为准，则是 `2048 x 2048`。

技术实现时建议：

```text
1. 当前本地 benchmark 如果来自最新 run_all.cpp，则按 512 x 512 调试。
2. 报告中注明 README/PDF 与 run_all.cpp 的不一致。
3. 最终性能判断以 grading 时使用的官方 harness 为准。
```

### 补充 3：数据结构转换是否计时，不同 kernel 规则不同

`graph` 和 `filter_gradient` 属于 data structure optimization 类型，原始要求允许设计新数据结构，并且 conversion 可以在 timed kernel 外单独调用，因此 conversion time 不计入最终 kernel runtime。

但是 `sparse_spmm` 不同：  
其函数签名不能修改，如果需要做数据转换或重排，必须放在 timed kernel 内，因此 conversion cost 会计入 runtime。

因此技术实现时应区分：

```text
graph / filter_gradient:
    conversion 放在 run_benchmark 之前，通常不计时。

sparse_spmm:
    conversion 如果需要，必须在 stu_sparse_spmm 内完成，会被计时。
```

### 补充 4：`k_best = 20` 实际是取平均，不是取 best

当前 `run_all.cpp` 中变量名可能叫 `k_best`，但实现逻辑是：

```cpp
avg_time += elapsed;
return avg_time / k_best;
```

因此它实际计算的是 20 次运行的平均时间，而不是最小时间 / best time。  
报告中如果描述测量方法，应写成：

```text
Each benchmark is executed 20 times and the average runtime is reported.
```

不要写成：

```text
The best runtime among 20 runs is reported.
```


---

## 1. 本次需要修改的代码文件

建议重点修改以下文件：

```text
include/bench.h
include/bitwise.h
include/blackscholes.h
include/filter_gradient.h
include/graph.h
include/grff.h
include/image_proc.h
include/matmul.h
include/relu.h
include/sparse_spmm.h
include/trace_replay.h
src/main/run_all.cpp
src/main/single_bench.cpp    # 如果用于正式测量，也建议同步修改
report source / report.pdf   # 修改实验表格和说明口径
```

---

## 2. Header 常量修改

### 2.1 总体要求

每个 kernel header 中保留原来的 `BASELINE_*`，并新增对应的：

```cpp
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_*{...};
```

注意：如果 `run_all.cpp` 已经引用了这些常量，但 header 没有定义，会出现类似错误：

```text
NAIVE_SPEEDUP_LOWER_BOUND_RELU was not declared in this scope
```

---

### 2.2 逐文件修改内容

#### `include/bitwise.h`

```cpp
const std::chrono::nanoseconds BASELINE_BITWISE{250000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_BITWISE{8.00};
```

#### `include/blackscholes.h`

```cpp
const std::chrono::nanoseconds BASELINE_BLACKSCHOLES{4800000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_BLACKSCHOLES{1.35};
```

#### `include/filter_gradient.h`

```cpp
inline constexpr std::chrono::nanoseconds BASELINE_FILTER_GRADIENT{25000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_FILTER_GRADIENT{1.45};
```

#### `include/graph.h`

```cpp
const std::chrono::nanoseconds BASELINE_GRAPH{5000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_GRAPH{2.50};
```

#### `include/grff.h`

```cpp
const std::chrono::nanoseconds BASELINE_GRFF{8500000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_GRFF{3.15};
```

#### `include/image_proc.h`

```cpp
const std::chrono::nanoseconds BASELINE_IMAGE_PROC{43000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_IMAGE_PROC{1.73};
```

#### `include/matmul.h`

```cpp
const std::chrono::nanoseconds BASELINE_MATMUL{88000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_MATMUL{2.45};
```

#### `include/relu.h`

```cpp
const std::chrono::nanoseconds BASELINE_RELU{550000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_RELU{2.50};
```

#### `include/sparse_spmm.h`

```cpp
const std::chrono::nanoseconds BASELINE_SPARSE_SPMM{116000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_SPARSE_SPMM{1.40};
```

#### `include/trace_replay.h`

```cpp
const std::chrono::nanoseconds BASELINE_TRACE_REPLAY{3400000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_TRACE_REPLAY{1.75};
```

---

## 3. 最新 basic 目标表

| Kernel | Baseline time ns | Lower bound `LB` | Basic 目标 |
|---|---:|---:|---|
| Bitwise | 250,000 | 8.00 | `T_naive / T_stu > 8.00` |
| Black-Scholes | 4,800,000 | 1.35 | `T_naive / T_stu > 1.35` |
| Filter Gradient | 25,000,000 | 1.45 | `T_naive / T_stu > 1.45` |
| Graph | 5,000,000 | 2.50 | `T_naive / T_stu > 2.50` |
| GRFF | 8,500,000 | 3.15 | `T_naive / T_stu > 3.15` |
| Image Proc | 43,000,000 | 1.73 | `T_naive / T_stu > 1.73` |
| MatMul | 88,000,000 | 2.45 | `T_naive / T_stu > 2.45` |
| ReLU | 550,000 | 2.50 | `T_naive / T_stu > 2.50` |
| Sparse SpMM | 116,000,000 | 1.40 | `T_naive / T_stu > 1.40`，注意 README/PDF 标称 2048×2048，但当前 `run_all.cpp` 可能实际使用 512×512 |
| Trace Replay | 3,400,000 | 1.75 | `T_naive / T_stu > 1.75` |

---

## 4. `include/bench.h` 修改

### 4.1 `bench_t` 结构体增加 lower bound 字段

原始结构可能类似：

```cpp
typedef struct {
    std::string description;
    lab_test_func tfunc;
    lab_test_func naiveFunc;
    lab_check_func checkFunc;
    void *args;
    void *ref_args;
    std::chrono::nanoseconds baseline_time;
} bench_t;
```

修改为：

```cpp
typedef struct {
    std::string description;
    lab_test_func tfunc;        // Student implementation
    lab_test_func naiveFunc;    // Naive implementation
    lab_check_func checkFunc;   // Correctness checker
    void *args;                 // Student benchmark context
    void *ref_args;             // Naive/reference context
    std::chrono::nanoseconds baseline_time;
    double naive_speedup_lower_bound = 1.0;
} bench_t;
```

---

### 4.2 保留 baseline speedup 计算函数

建议保留如下函数：

```cpp
inline double calculate_speedup(std::chrono::nanoseconds measured_time,
                                std::chrono::nanoseconds baseline_time) {
    const auto measured_count = measured_time.count();
    const auto baseline_count = baseline_time.count();

    if (measured_count <= 0 || baseline_count <= 0) {
        throw std::invalid_argument(
            "calculate_speedup: measured_time and baseline_time must be positive.");
    }

    return static_cast<double>(baseline_count) /
           static_cast<double>(measured_count);
}
```

用途：

```cpp
// baseline / stu
calculate_speedup(stu_time, bench.baseline_time);

// naive / stu
calculate_speedup(stu_time, naive_time);
```

注意参数顺序是：

```cpp
calculate_speedup(measured_time, baseline_time)
```

所以计算 `T_naive / T_stu` 时，应写：

```cpp
calculate_speedup(stu_time, naive_time);
```

---

### 4.3 增加 adjusted naive speedup 计算函数

用于 bonus GM：

```cpp
inline double calculate_adjusted_naive_speedup(
    std::chrono::nanoseconds naive_time,
    std::chrono::nanoseconds stu_time,
    double naive_speedup_lower_bound) {

    const auto naive_count = naive_time.count();
    const auto stu_count = stu_time.count();

    if (naive_count <= 0 || stu_count <= 0 ||
        naive_speedup_lower_bound <= 0.0) {
        throw std::invalid_argument(
            "calculate_adjusted_naive_speedup: inputs must be positive.");
    }

    return static_cast<double>(naive_count) /
           (static_cast<double>(stu_count) * naive_speedup_lower_bound);
}
```

建议增加 overload：

```cpp
inline double calculate_adjusted_naive_speedup(
    const bench_t &bench,
    std::chrono::nanoseconds naive_time,
    std::chrono::nanoseconds stu_time) {

    return calculate_adjusted_naive_speedup(
        naive_time,
        stu_time,
        bench.naive_speedup_lower_bound);
}
```

---

### 4.4 修改 geometric mean 计算

原始 bonus GM 可能基于：

```cpp
speedup = baseline / stu;
```

现在需要改为：

```cpp
speedup = naive / (stu * LB);
```

建议增加如下版本：

```cpp
inline double calculate_geometric_mean_speedup(
    const std::vector<std::chrono::nanoseconds> &naive_times,
    const std::vector<std::chrono::nanoseconds> &stu_times,
    const std::vector<bench_t> &benchmarks) {

    if (naive_times.size() != benchmarks.size() ||
        stu_times.size() != benchmarks.size()) {
        throw std::invalid_argument(
            "calculate_geometric_mean_speedup: input size mismatch.");
    }

    std::vector<double> speedups;
    speedups.reserve(benchmarks.size());

    for (size_t i = 0; i < benchmarks.size(); ++i) {
        speedups.push_back(calculate_adjusted_naive_speedup(
            benchmarks[i], naive_times[i], stu_times[i]));
    }

    return calculate_geometric_mean_speedup(speedups);
}
```

---

## 5. `src/main/run_all.cpp` 修改

### 5.1 输出表头增加 naive 和 student 时间

建议表头改为：

```cpp
std::cout << std::left << std::setw(25) << "Benchmark"
          << std::setw(12) << "Status"
          << std::right << std::setw(18) << "Naive (ns)"
          << std::setw(18) << "Stu (ns)"
          << std::setw(14) << "vs Naive"
          << std::setw(16) << "vs Baseline"
          << '\n';
```

---

### 5.2 每个 benchmark 先测 naive，再测 student

建议实现一个平均测量函数：

```cpp
const auto measure_average_time = [&](lab_test_func func,
                                      void *ctx,
                                      const std::string &label) {
    std::chrono::nanoseconds avg_time{0};

    for (int i = 0; i < k_best; ++i) {
        flush_cache();
        const auto elapsed = measure_time([&] {
            func(ctx);
        });

        avg_time += elapsed;

        debug_log("DEBUG: {} {}-th measurement: {} ns\n",
                  label, i, static_cast<long long>(elapsed.count()));
    }

    return avg_time / static_cast<int>(k_best);
};
```

先测 naive：

```cpp
const auto naive_time = measure_average_time(
    bench.naiveFunc,
    bench.ref_args,
    bench.description + " naive");
```

如果 student 尚未实现：

```cpp
if (bench.tfunc == nullptr || bench.args == nullptr) {
    std::cout << "TODO" << ...;
    return;
}
```

再测 student：

```cpp
const auto stu_time = measure_average_time(
    bench.tfunc,
    bench.args,
    bench.description + " stu");
```

---

### 5.3 correctness check 仍然优先

保持如下逻辑：

```cpp
const bool correct = bench.checkFunc(
    bench.args,
    bench.ref_args,
    bench.naiveFunc);
```

如果失败：

```cpp
std::cout << "FAILED" << ...;
std::cout << " Error: Results do not match naive implementation!\n";
gm_enable = false;
return;
```

注意：任何 kernel 如果 correctness 失败，性能结果无效。

---

### 5.4 计算 `vs Naive` 和 `vs Baseline`

通过 correctness 后：

```cpp
const double naive_speedup = calculate_speedup(stu_time, naive_time);
const double baseline_speedup = calculate_speedup(bench, stu_time);
```

含义：

```text
naive_speedup    = T_naive / T_stu
baseline_speedup = T_baseline / T_stu
```

输出示例：

```cpp
std::cout << "PASSED"
          << std::right
          << std::setw(18) << naive_time.count()
          << std::setw(18) << stu_time.count()
          << std::setw(13) << std::fixed << std::setprecision(3)
          << naive_speedup << "x"
          << std::setw(15) << baseline_speedup << "x";
```

---

### 5.5 `(SLOWER)` 判断逻辑修改

旧逻辑可能是：

```cpp
stu_time > baseline_time
```

或：

```cpp
stu_time > 1.1 * baseline_time
```

现在应改为：

```cpp
if (naive_speedup < bench.naive_speedup_lower_bound) {
    std::cout << " (SLOWER)";
}
```

含义：

```text
if T_naive / T_stu < LB:
    mark as SLOWER
```

---

## 6. 每个 kernel 的 `run_benchmark` 注册方式修改

每个 `run_benchmark({...})` 的最后都要增加对应的 lower bound 参数。

---

### 6.1 ReLU 示例

原始：

```cpp
run_benchmark({"ReLU",
               stu_relu_wrapper,
               naive_relu_wrapper,
               relu_check,
               &relu_args_stu,
               &relu_args_ref,
               BASELINE_RELU});
```

修改为：

```cpp
run_benchmark({"ReLU",
               stu_relu_wrapper,
               naive_relu_wrapper,
               relu_check,
               &relu_args_stu,
               &relu_args_ref,
               BASELINE_RELU,
               NAIVE_SPEEDUP_LOWER_BOUND_RELU});
```

---

### 6.2 所有 kernel entry 示例

#### Black-Scholes

```cpp
run_benchmark({"Black-Scholes",
               stu_BlkSchls_wrapper,
               naive_BlkSchls_wrapper,
               BlkSchls_check,
               &black_args_stu,
               &black_args_ref,
               BASELINE_BLACKSCHOLES,
               NAIVE_SPEEDUP_LOWER_BOUND_BLACKSCHOLES});
```

#### Sparse SpMM

```cpp
run_benchmark({"Sparse SpMM",
               stu_sparse_spmm_wrapper,
               naive_sparse_spmm_wrapper,
               sparse_spmm_check,
               &sparse_args_stu,
               &sparse_args_ref,
               BASELINE_SPARSE_SPMM,
               NAIVE_SPEEDUP_LOWER_BOUND_SPARSE_SPMM});
```


> **注意：Sparse SpMM 的 size 需要特别检查。**  
> README/PDF 中写的是 `2048 x 2048`，但当前最新 `run_all.cpp` 里初始化可能是 `512 x 512`。  
> 技术同学应优先检查本地实际 `initialize_spmm(...)` 的参数，并在报告中注明最终实验采用的 size。最终 grading 以官方 harness 为准。


#### ReLU

```cpp
run_benchmark({"ReLU",
               stu_relu_wrapper,
               naive_relu_wrapper,
               relu_check,
               &relu_args_stu,
               &relu_args_ref,
               BASELINE_RELU,
               NAIVE_SPEEDUP_LOWER_BOUND_RELU});
```

#### Bitwise

```cpp
run_benchmark({"Bitwise",
               stu_bitwise_wrapper,
               naive_bitwise_wrapper,
               bitwise_check,
               &bitwise_args_stu,
               &bitwise_args_ref,
               BASELINE_BITWISE,
               NAIVE_SPEEDUP_LOWER_BOUND_BITWISE});
```

#### MatMul

```cpp
run_benchmark({"MatMul",
               stu_matmul_wrapper,
               naive_matmul_wrapper,
               matmul_check,
               &matmul_args_stu,
               &matmul_args_ref,
               BASELINE_MATMUL,
               NAIVE_SPEEDUP_LOWER_BOUND_MATMUL});
```

#### Trace Replay

```cpp
run_benchmark({"Trace Replay",
               stu_trace_replay_wrapper,
               naive_trace_replay_wrapper,
               trace_replay_check,
               &trace_args_stu,
               &trace_args_ref,
               BASELINE_TRACE_REPLAY,
               NAIVE_SPEEDUP_LOWER_BOUND_TRACE_REPLAY});
```

#### Graph

```cpp
run_benchmark({"Graph",
               stu_graph_wrapper,
               naive_graph_wrapper,
               graph_check,
               &graph_args_stu,
               &graph_args_ref,
               BASELINE_GRAPH,
               NAIVE_SPEEDUP_LOWER_BOUND_GRAPH});
```

#### GRFF

```cpp
run_benchmark({"GRFF",
               stu_grff_wrapper,
               naive_grff_wrapper,
               grff_check,
               &grff_args_stu,
               &grff_args_ref,
               BASELINE_GRFF,
               NAIVE_SPEEDUP_LOWER_BOUND_GRFF});
```

#### Image Proc

```cpp
run_benchmark({"Image Proc",
               stu_image_proc_wrapper,
               naive_image_proc_wrapper,
               image_proc_check,
               &image_args_stu,
               &image_args_ref,
               BASELINE_IMAGE_PROC,
               NAIVE_SPEEDUP_LOWER_BOUND_IMAGE_PROC});
```

#### Filter Gradient

```cpp
run_benchmark({"Filter Gradient",
               stu_filter_gradient_wrapper,
               naive_filter_gradient_wrapper,
               filter_gradient_check,
               &filter_gradient_args_stu,
               &filter_gradient_args_ref,
               BASELINE_FILTER_GRADIENT,
               NAIVE_SPEEDUP_LOWER_BOUND_FILTER_GRADIENT});
```

---

## 7. Student context 和 reference context 必须分开

很多 kernel 会 in-place 修改输入，因此必须使用独立的 student args 和 reference args。

推荐写法：

```cpp
xxx_args xxx_args_ref;
initialize_xxx(&xxx_args_ref, size, seed);

xxx_args xxx_args_stu;
initialize_xxx(&xxx_args_stu, size, seed);

run_benchmark({...,
               &xxx_args_stu,
               &xxx_args_ref,
               ...});
```

不要写成：

```cpp
run_benchmark({...,
               &xxx_args,
               &xxx_args,
               ...});
```

原因：student function 可能先修改输入，导致 reference context 被污染，进而影响 naive timing 或 correctness check。


---

## 7.1 数据结构转换是否计时的特殊说明

不同 kernel 的 conversion timing 规则不同，不能混淆。

### `graph` / `filter_gradient`

这两个属于 data structure optimization 类型。原始要求允许为优化版本设计新的数据结构，也允许 conversion function 单独调用。  
因此推荐做法是：

```text
1. 初始化 naive/reference 数据。
2. 初始化 student 数据。
3. 在 run_benchmark 之前调用 conversion。
4. benchmark 只测 optimized kernel 本身。
```

示例：

```cpp
convert_graph_to_csr(graph_args_stu.graph_csr, graph_args_stu.graph);

run_benchmark({"Graph",
               stu_graph_wrapper,
               naive_graph_wrapper,
               graph_check,
               &graph_args_stu,
               &graph_args_ref,
               BASELINE_GRAPH,
               NAIVE_SPEEDUP_LOWER_BOUND_GRAPH});
```

### `sparse_spmm`

`sparse_spmm` 不属于这个特殊例外。它的函数签名不能修改。  
如果优化策略需要转换 dense matrix layout、重排 CSR 数据或构造临时 block layout，这些操作必须在 `stu_sparse_spmm` 内完成，因此会被计入 runtime。

```text
graph / filter_gradient:
    conversion 可以放在 timed kernel 外。

sparse_spmm:
    conversion 必须放在 timed kernel 内。
```


---

## 8. TODO block 处理

很多原始或更新后的 `run_all.cpp` 中会保留两段逻辑：

1. 已实现 student wrapper 的 block，但被注释。
2. `nullptr` 的 TODO block，用于未实现时占位。

当某个 kernel 的 student implementation 已经完成后，需要：

```text
1. 打开 student wrapper block。
2. 注释或删除 nullptr TODO block。
3. 确保最后一个参数包含对应的 NAIVE_SPEEDUP_LOWER_BOUND_*。
```

例如 ReLU：

```cpp
relu_args relu_args_stu;
initialize_relu(&relu_args_stu, relu_size, seed);

run_benchmark({"ReLU",
               stu_relu_wrapper,
               naive_relu_wrapper,
               relu_check,
               &relu_args_stu,
               &relu_args_ref,
               BASELINE_RELU,
               NAIVE_SPEEDUP_LOWER_BOUND_RELU});

// 删除或注释掉 nullptr TODO block
```

---

## 9. Bonus 部分修改

### 9.1 旧公式

原始 bonus 每个 kernel 的 speedup 是：

```text
T_baseline / T_bonus
```

### 9.2 新公式

现在改为：

```text
T_naive / (T_bonus * LB)
```

也就是：

```text
(T_naive / T_bonus) / LB
```


> **注意：这里是 bonus / normalized GM 口径。**  
> basic part 的逐 kernel 目标仍然是 `T_naive / T_stu > LB`。  
> 不要把最终打印的 lower-bound-normalized GM 当作 basic 的唯一判断标准。


---

### 9.3 代码实现

如果原来 GM 使用：

```cpp
const double geometric_mean_speedup =
    calculate_geometric_mean_speedup(gm_speedups);
```

现在建议改为：

```cpp
const double geometric_mean_speedup =
    calculate_geometric_mean_speedup(
        gm_naive_times,
        gm_stu_times,
        gm_benchmarks);
```

每个 benchmark passed 后收集：

```cpp
gm_naive_times.push_back(naive_time);
gm_stu_times.push_back(stu_time);
gm_benchmarks.push_back(bench);
```

注意：只有 correctness passed 的 kernel 才应进入 GM 计算。

---

## 10. `src/main/single_bench.cpp` 修改建议

如果 `single_bench.cpp` 只是开发调试工具，可以暂时不完全同步。

但如果它的输出会被用于 report 或最终性能判断，建议同步修改：

```text
1. bench_t 初始化时加入 NAIVE_SPEEDUP_LOWER_BOUND_*。
2. 同时测 naive 和 stu。
3. 输出 T_naive、T_stu、T_naive / T_stu、LB。
4. 判断是否达标时使用 T_naive / T_stu > LB。
```

不要再只看：

```text
T_stu <= BASELINE
```

---

## 11. Report 实验表格修改

### 11.1 Basic performance table

建议改成：

| Kernel | `T_naive` ns | `T_stu` ns | `T_naive / T_stu` | LB | Basic target met? | `T_baseline / T_stu` |
|---|---:|---:|---:|---:|---|---:|
| Bitwise | ... | ... | ... | 8.00 | Yes/No | ... |
| Black-Scholes | ... | ... | ... | 1.35 | Yes/No | ... |
| Filter Gradient | ... | ... | ... | 1.45 | Yes/No | ... |
| Graph | ... | ... | ... | 2.50 | Yes/No | ... |
| GRFF | ... | ... | ... | 3.15 | Yes/No | ... |
| Image Proc | ... | ... | ... | 1.73 | Yes/No | ... |
| MatMul | ... | ... | ... | 2.45 | Yes/No | ... |
| ReLU | ... | ... | ... | 2.50 | Yes/No | ... |
| Sparse SpMM | ... | ... | ... | 1.40 | Yes/No | ... |
| Trace Replay | ... | ... | ... | 1.75 | Yes/No | ... |

---

### 11.2 推荐替换报告表述

不建议再写：

```text
The optimized implementation is considered successful if T_stu <= T_baseline.
```

建议改成：

```text
After the benchmark update, the primary performance metric for the basic part is the speedup over the naive implementation, defined as T_naive / T_stu. Each kernel has a corresponding lower-bound value LB, and the optimization target is to make T_naive / T_stu exceed LB. We still report T_baseline / T_stu as an auxiliary reference.
```

---

### 11.3 Bonus report table

如果做 bonus，建议增加：

| Kernel | `T_naive` ns | `T_bonus` ns | LB | Raw speedup `T_naive/T_bonus` | Adjusted multiplier `T_naive/(T_bonus*LB)` |
|---|---:|---:|---:|---:|---:|
| Bitwise | ... | ... | 8.00 | ... | ... |
| Black-Scholes | ... | ... | 1.35 | ... | ... |
| Filter Gradient | ... | ... | 1.45 | ... | ... |
| Graph | ... | ... | 2.50 | ... | ... |
| GRFF | ... | ... | 3.15 | ... | ... |
| Image Proc | ... | ... | 1.73 | ... | ... |
| MatMul | ... | ... | 2.45 | ... | ... |
| ReLU | ... | ... | 2.50 | ... | ... |
| Sparse SpMM | ... | ... | 1.40 | ... | ... |
| Trace Replay | ... | ... | 1.75 | ... | ... |

GM 公式写成：

```text
GM = (Π_i T_naive,i / (T_bonus,i * LB_i))^(1/n)
```

---

## 12. 验收标准

### 12.1 编译验收

运行：

```bash
cmake -S . -B build
cmake --build build -j
```

必须没有以下错误：

```text
NAIVE_SPEEDUP_LOWER_BOUND_RELU was not declared in this scope
NAIVE_SPEEDUP_LOWER_BOUND_BITWISE was not declared in this scope
...
```

如果出现，说明对应 header 没有补 lower bound 常量。

---

### 12.2 运行验收

运行：

```bash
./build/run_all
```


注意：如果代码中变量名为 `k_best = 20`，当前实现仍然是对 20 次运行取平均值，而不是取最小值。  
报告中应描述为：

```text
The benchmark is executed 20 times and the average runtime is reported.
```


输出应包含类似列：

```text
Benchmark                 Status        Naive (ns)          Stu (ns)      vs Naive     vs Baseline
```

每个已实现 kernel 应输出：

```text
PASSED
```

并且有两个 speedup：

```text
vs Naive
vs Baseline
```

如果 `vs Naive < LB`，应出现：

```text
(SLOWER)
```

---

### 12.3 Correctness 验收

任何 kernel 如果输出：

```text
FAILED
Error: Results do not match naive implementation!
```

则该 kernel 性能结果无效，必须先修 correctness。

---

### 12.4 Performance 验收

对每个 kernel 计算：

```text
speedup_over_naive = T_naive / T_stu
```

并和 lower bound 比较：

```text
speedup_over_naive > LB
```

示例：

```text
ReLU:
T_naive = 1,300,000 ns
T_stu   =   480,000 ns
speedup = 1,300,000 / 480,000 = 2.708
LB      = 2.50
=> pass
```

---

## 13. 技术修改任务清单

```text
[ ] 1. 在 10 个 include/*.h 中补充 NAIVE_SPEEDUP_LOWER_BOUND_* 常量。
[ ] 2. 修改 include/bench.h：bench_t 增加 naive_speedup_lower_bound。
[ ] 3. 修改 include/bench.h：增加 calculate_adjusted_naive_speedup。
[ ] 4. 修改 include/bench.h：GM 改为 naive / (stu * LB)。
[ ] 5. 修改 src/main/run_all.cpp：每个 benchmark 同时测 naive 和 stu。
[ ] 6. 修改 src/main/run_all.cpp：输出 Naive(ns)、Stu(ns)、vs Naive、vs Baseline。
[ ] 7. 修改 src/main/run_all.cpp：SLOWER 判断改为 vs Naive < LB。
[ ] 8. 修改 src/main/run_all.cpp：每个 run_benchmark entry 最后加入对应 LB 常量。
[ ] 9. 打开已完成 kernel 的 student block，注释掉 nullptr TODO block。
[ ] 10. 确保 student args 和 reference args 分离，避免 in-place 污染。
[ ] 11. 如使用 single_bench 记录结果，也同步上述逻辑。
[ ] 12. 修改 report 表格：主指标改为 T_naive / T_stu 和 LB。
[ ] 13. 如做 bonus，GM 公式改为 T_naive / (T_bonus * LB)。
[ ] 14. 明确区分 basic pass/fail 与 lower-bound-normalized GM：basic 逐 kernel 看 T_naive / T_stu > LB。
[ ] 15. 检查 sparse_spmm 的实际 input size：README/PDF 是 2048 x 2048，当前 run_all.cpp 可能是 512 x 512。
[ ] 16. 区分 conversion timing：graph/filter_gradient 可在 timed kernel 外转换，sparse_spmm 转换必须计时。
[ ] 17. 报告中说明当前 benchmark 是 20 次平均时间，不是 20 次中的 best/min 时间。
```

---

## 14. 最终总结

本次最新修改的本质不是要求重写优化算法，而是修改性能评价口径：

```text
原始 PDF basic:
    主要看 T_stu <= T_baseline

最新 basic:
    主要看 T_naive / T_stu > LB

原始 PDF bonus:
    GM 使用 T_baseline / T_bonus

最新 bonus:
    GM 使用 T_naive / (T_bonus * LB)
```

因此技术实现上，关键是让 benchmark harness 能够同时测出 `T_naive` 和 `T_stu`，并把每个 kernel 的 lower bound 纳入 basic 判断和 bonus GM 计算。

同时需要特别注意：

```text
1. basic pass/fail 是逐 kernel 判断 T_naive / T_stu > LB；
   GM = T_naive / (T_stu * LB) 是 lower-bound-normalized aggregate 指标。

2. sparse_spmm 的测试规模需要检查本地 run_all.cpp：
   README/PDF 标称 2048 x 2048，但当前 run_all.cpp 可能实际使用 512 x 512。

3. graph/filter_gradient 的数据结构 conversion 可以放在 timed kernel 外；
   sparse_spmm 的 conversion 如果存在，必须放在 timed kernel 内。

4. 当前 run_all.cpp 的 k_best = 20 实际表示 20 次平均时间，
   不是 20 次中的最优时间。
```
