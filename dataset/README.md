# TNSE-SimSimple-v1 — 轻量级模拟模型描述符数据集

**用途：** 为 TNSE 的六组实验（9 个 Figure）提供统一、可复现、低复杂度的数据接口。该数据集包含**真实生成并保存的 11 份基础描述符库**、5 份 200-round 动态参考事件、资源参数库，以及按需加载任意实验配置的 Python API。

> **真实性声明：** 所有模型描述符都是合成的，**没有**从实际预训练/量化/部署的 DNN 中提取，也**没有**真实网络时延观测。数据用于验证算法行为与模拟系统开销，不能宣称真实部署准确率。

> **严谨性声明：** 生成器仅按预先固定的角度扰动规则生成 normal/corrupt 模型描述符。数据集未运行 Proposed 或任何 baseline，也没有为保证 Proposed 获胜而筛选种子或修改性能指标。

## 1. 目录结构

```text
TNSE_Simple_Dataset/
├── README.md
├── config.json
├── dataset_summary.csv         # 729 个可重现的“逻辑实验组”索引
├── load_dataset.py             # 通用加载 + 资源 + 动态事件函数
├── materialize_group.py        # 有需要时输出物理 vectors.npz
├── generate_dataset.py         # 原始可复现数据生成器
├── validate_dataset.py         # 全量质量检查
├── requirements.txt
├── MANIFEST.sha256
├── banks/
│   ├── bank_H_00.npz            # H=0.0
│   ├── ...
│   └── bank_H_10.npz            # H=1.0
├── dynamic/
│   ├── reference_cr_01.npz     # 200轮，1%经历损坏事件的模型
│   ├── ...
│   └── reference_cr_05.npz
├── resources/
│   └── reference_profiles.npz
├── examples/
│   └── n_3000/corruption_03/dim_30/
│       ├── vectors.npz
│       ├── row_metadata.csv
│       └── metadata.json
└── validation/
    └── VALIDATION_REPORT.md
```

**特别说明：729 组是可通过 API 重现的参数组合，而不是 729 个在磁盘中逐个保存的 `vectors.npz`。** 为避免生成大量重复文件（特别是 11 档 H × 729 组），采用 TMC 近似的接口，由已生成的基础库确定性抽取并构造所需组。真正的 float32 描述符已经存储在 `banks/*.npz`，不是空配置。用户如需某组物理文件，可调用 `materialize_group.py`。

## 2. 规模和实验参数

| 参数 | 可用值 | 默认 |
|---|---|---|
| 模型实例数量 `N` | 1000, 1500, ..., 5000（9点） | 3000 |
| 模型损坏比例 `cr_percent` | 1.0, 1.5, ..., 5.0%（9点） | 3.0% |
| 描述符维度 `d` | 10, 15, ..., 50（9点） | 30 |
| 合法异构强度 `H` | 0.0, 0.1, ..., 1.0（11点） | 0.5 |
| 相似度阈值 `tau_lower` | 0.990,0.991,...,0.999,0.9995（11点） | 0.997 |
| 候选异常比例 `q_C` | 0%–50%（常用 5% 步长） | 10% |
| 候选通信失败率 `p_fail` | 0%,5%,10%,15%,20% | 5% |
| 资源异构强度 `sigma` | 0.0–1.0（支持任意指定） | 0.5 |
| 预算比例 `beta_B` | 5%,10%,15%,20%,25% | 10% |
| Long-Term rounds | 200 | 200 |

对应六组：Detection 2图；Computation/Communication 2图；Threshold 1图；Ground-Truth 1图；Joint Decision 2图；Long-Term 1图，共9 Figure。图种类不限定；各算法配色必须全论文统一，图类型依指标选取。

## 3. 模型描述符与四类样本

每个 `bank_H_XX.npz` **实际存有9000条** 50D float32 描述符：

| `subtype_label` | 名称（保留 TMC 字段兼容） | 单库条数 | 含义 |
|---:|---|---:|---|
| 0 | `common_normal` | 4,800 | 常规合法部署差异 |
| 1 | `high_personalized_normal` | 1,200 | 高合法异构、可接近或跨越阈值的正常实例 |
| 2 | `severe_corrupt` | 2,100 | 显著异常模型状态 |
| 3 | `mild_corrupt` | 900 | 边界轻度损坏，存在可能漏检样本 |

每个 H 库共6000正常、3000损坏。11个 H 库合计99,000条**已保存的数据行**，但相同来源 ID 在不同 H 下重复出现，不能把它们当作99,000个独立模型实例。`H` 只影响正常样本的合法差异，**损坏样本随 H 不改变**，便于控制变量。

正常扰动采用高斯及局部块相关方向；损坏扰动采用较稀疏的特征方向；通过单位球面上的不同 angular magnitude 控制相似度。`high_personalized_normal` 在论文可称 high-heterogeneity normal，但磁盘标签名不变。该机制模拟描述符空间中的差异，不对应真实执行的量化、微调或模型文件损坏。

对于一个逻辑组，默认普通正常:高异构正常比例约80:20，显著损坏:轻度损坏约80:20。小样本组由于整数配额取整会有微小差异；实际数量以返回值 `counts` 为准。

### 3.1 维度处理

数据中的 canonical 维度为50。`load_group()` 提供 `d={10,15,...,50}`，使用相同 canonical 子空间，经 deterministic re-orthogonalization 构造对应维度的单位描述符，**使同一模型在不同 d 下的角度偏移强度保持可比**。它不是声称实际模型输出了不同大小的特征张量，也不是简单将首 d 维截断后未经处理地计算阈值。

### 3.2 Normal–Normal Pairwise 检查

TNSE 的 GT 算法比较的是**候选模型之间**的相似度，而不只是“模型与原型之间”的相似度。生成器和校验器专门检查普通正常–普通正常的真实 pairwise cosine，以及 K=50、24异常+26正常的 majority 案例。**Theorem 1 的严格条件不能仅凭样本 `binary_label==0` 保证**，尤其 H 很高或 mild corruption 与正常分布重叠时。论文应如实报告 GT 无法形成或误纳入坏模型的次数。

## 4. 读取逻辑实验组：推荐方式

在 `TNSE_Simple_Dataset` 文件夹直接执行：

```python
from load_dataset import load_group

g = load_group(
    dataset_root='.', n=3000,
    corruption_rate_percent=3.0,
    dim=30, H=0.5, seed=2026
)

Q = g['X']                       # shape (3000,30), float32
labels = g['binary_label']      # 0 normal, 1 corrupt
subtypes = g['subtype_label']    # 0/1/2/3
prototype = g['prototype']
cosines = g['cosine_to_prototype']
print(g['counts'])
```

在外部 TNSE 项目中建议把解压后的文件夹改名为 `dataset/`，并从 `dataset.load_dataset` 导入（`dataset/` 中可加入空的 `__init__.py`，或把 `dataset/` 加入 `sys.path`）。**不需要**在 `dataset/n_1500/corruption_015/dim_30/` 找物理文件。上述组由基础库确定性生成。

返回值包含与 TMC 相似的 `X`, `binary_label`, `subtype_label`, `cosine_to_prototype`, `prototype`, `tau_lower`, `tau_upper`，另含 `sample_id`, `N`, `dimension`, `heterogeneity_H`, `counts`, `seed`。

如果后续代码坚持直接使用物理 NPZ 文件，可执行：

```bash
python materialize_group.py --n 1500 --cr 1.5 --d 30 --H 0.5 \
    --output-dir ./my_group
```

产生 `vectors.npz`, `row_metadata.csv`, `metadata.json`。`examples/` 里提供了一组已经物理保存的例子。

**损坏比例取整**：如 N=1500、cr=1.5%，理论22.5个，按 half-up 取23个；`cr_percent` 为名义比例，真实比例可由 `binary_label.mean()` 得到。

## 5. GT 候选集构造

```python
from load_dataset import load_group, draw_candidate_set

g = load_group('.', n=3000, corruption_rate_percent=3, dim=30)
candidate_indices, q_realized = draw_candidate_set(
    g, K=50, q_C=0.48, seed=2026,
    use_severe_only=True, use_common_only=True
)
assert (g['binary_label'][candidate_indices] == 1).sum() == 24
```

这一抽样函数使用真实标签**仅用于构造预设比例的实验 workload**。生成 candidate 集合后，TNSE Proposed 不允许读取当前标签进行 GT 判断或 risk estimation。用于检查 strict-majority 定理时，可设置 `use_common_only=True, use_severe_only=True` 来隔离描述符可分性假设；通常鲁棒性测试则应包含所有 subtype。

**重要整数问题**：K=50时，5%对应2.5个异常，无法每次恰好5%。`draw_candidate_set()` 用可复现的随机取整，使跨多次试验的**期望**比例等于名义设定，且返回每次实际比例 `q_realized`；不得声称单次 sample 精确5%。48%和50%等边界仍是精确整数。

## 6. Long-Term 的模拟数据（200轮）

数据集包含 `dynamic/reference_cr_01.npz` ... `05.npz` 作为 seed=2026 的参考；其它随机种子通过函数生成：

```python
from load_dataset import generate_dynamic_trace, load_dynamic_case

tr = generate_dynamic_trace(cr_percent=3, seed=2027,
                            n=1000, rounds=200)
case = load_dynamic_case('.', cr_percent=3, seed=2027,
                         n=1000, d=30, H=0.5, rounds=200)

# case['normal_X']: [1000,30]
# case['corrupted_event_X']: [30,30]
# case['event_model_ids']: the 30 affected identities
# case['corrupted_mask']: [200,1000]
# case['fail_uniform']: [200,1000]
```

`cr_dynamic` 的含义：在整段200轮观察期内，**出现过一次损坏事件的模型身份占比**，不是每轮瞬时损坏比例。每个受影响模型有预先生成的 onset 与 recovery 轮次；算法不能查看未来事件。`fail_uniform[t,i] < p_fail` 表示该模型该轮交互失败，可统一作用于所有算法。

在模拟器中，根据当前轮的 `corrupted_mask[t,i]` 决定实例显示正常或损坏描述符；真正发生损坏时，状态由外部事件改变，算法必须通过验证结果发现。不同算法必须共享同一 exogenous trace。未发现的事件按**未检测/删失**处理，不得从检测率分母中删掉。`A_i`, `h_i`, `t_last` 必须由算法从事件反馈逐轮更新，绝不从标签直接初始化。

## 7. Resource Heterogeneity

预生成参考基础资源参数：`resources/reference_profiles.npz`，并提供 stateless lookup：

```python
import numpy as np
from load_dataset import resource_profiles, resource_costs

r = resource_profiles(seed=2026, n=5000, rounds=200)
a = resource_costs(
    target_ids=np.array([100,101]),
    verifier_ids=np.array([3,7]),
    d=30, sigma=0.5, round_id=1, profiles=r
)
print(a['total_ms'])
```

资源中心值：**计算时间4 ms（d=30）、50 Mbps 有效带宽、网络传播延迟10 ms**；`sigma=0` 所有 pair 完全同质；增加 sigma 改变 log-normal 离散度。实际 `compute_ms`, `transfer_ms`, `network_ms`, `total_ms` 为模拟值。网络带宽影响的是传输时延，**不是传输字节数**。

TNSE 的一轮 makespan 应根据 per-verifier 累积验证时延（同一 verifier 串行、不同 verifier 并行）计算，而不是把所有模型的时延简单相加。`T_round_max=400ms` 是先验默认配置，实验必须记录 infeasible 情况，不能直接删除。

## 8. 六组/九张 Figure 的数据映射

| Figure | 核心变化 | 数据 API |
|---|---|---|
| D1 | N=1000..5000 每500，子图固定cr=1..5% | `load_group(N,cr,d=30,H=.5)` |
| D2 | cr=1.0..5.0每0.5%，子图固定N=1000..5000每1000 | `load_group(N,cr,d=30,H=.5)` |
| O1 | N=1000..5000每500，子图固定d=10,20,30,40,50 | `load_group()` + resource cost |
| O2 | d=10..50每5，子图固定N=1000..5000每1000 | `load_group()` + packet accounting |
| S1 | 11个 tau，子图固定cr=1..5% | 同批 `load_group()` 重用 descriptors |
| G1 | qC=0..50% + p_fail=0..20%，5子图 | `draw_candidate_set()` + fail draws |
| J1 | N=1000..5000每500，子图固定预算比5..25% | `load_group()` + `resource_costs()` |
| J2 | N=1000..5000每500，子图固定sigma五档 | `load_group()` + `resource_costs()` |
| L1 | T=200，5子图对应动态事件比例1..5% | `load_dynamic_case()` |

主文绘图以真实执行结果为准。**颜色、字体、marker 方案全文一致，但图形类型不必统一**：有序参数通常适合折线；子类/算法分类比较可以使用分组柱状图；构成占比可以用堆叠图；不要为图形美观更改原始数据。

## 9. 学术用途重要限制

1. `prototype` 是生成器的参考向量，**不是 Proposed 算法预先知道的 Ground-Truth**。仅可用于质量检查或明确定义为 oracle-reference 的诊断性 baseline。
2. `binary_label` / `subtype_label` 不得进入 Proposed 或 baseline 的风险得分、任务选择、GT 判断。它们用于受控构造和最终计算指标。
3. 所有 9 图均应使用相同种子集合，比较方法应在同一工作负载上配对运行；不得挑选好看的 seeds。
4. 图的采样点来自真实执行，不得由少数点插值作为新的测量数据。
5. D1/D2 中如果模型选择预算有限，Detection Accuracy 的分母必须定义清楚，不能把未验证实例自动当作正常；建议控制所有对比方法评估同一组已验证 targets，另研究调度算法的选择质量。
6. Joint Decision 与 Detection 是不同问题：前者看 `J` 和资源约束，后者看有实际验证结果时的准确性；不应为算法打分提前提供真实标签。
7. `lambda`, `alpha`, `rho`, `eta`, `K`, `M` 等为 TNSE **算法参数**，不是本数据集提供的外部 ground truth。
8. 如实验不满足预期，应检查实现/公平性与模型假设，绝不能针对 Proposed 反向修改模拟分布。
9. `dataset_summary.csv` 记录的是729个**虚拟可重构实验组**，而不是729个文件已物理保存。

## 10. 环境、复现与完整性

```bash
pip install -r requirements.txt
python validate_dataset.py
python generate_dataset.py   # 可选：重新生成基础文件；之后需再次运行 validate_dataset.py
```

验证结果见 `validation/VALIDATION_REPORT.md`。所有文件校验和见 `MANIFEST.sha256`。

**建议将解压文件夹统一重命名为 TNSE 代码根目录下的 `dataset/`，然后直接使用 `dataset/load_dataset.py`。**
