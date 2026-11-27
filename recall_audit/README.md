# RECALL-MM 修订公告分析

设计师知道产品大类、品名和外观。如果只能先查两个危险，历史召回能帮他选哪两个？找到提示后，再用公告确认该查哪一个部件。

## 阅读顺序

| 入口 | 回答的问题 |
|---|---|
| [01 公告核对、找规律与分组](01_公告核对与字段关联.ipynb) | 用 6,637 份修订公告找出每类常见的危险。再只用品名和外观分组，揭开产品大类和危险标签核对组间差别。 |
| [02 先查哪两个危险](02_危险分类评估.ipynb) | 在同一组五折比较 kNN、Gini 决策树和 XGBoost 等模型，也检查品名、外观和结构各自的作用。最后复核候选规律，并从「跌落」查到部件。 |
| [主工作流](../OUTLINE_NEO.md) | 按「先找规律、再评估提示、最后查部件」阅读，附有 A、B、C、家电子类与 D 的原实验结果。 |
| [English presentation outline](../PPT_OUTLINE_EN.md) | Field glossary and slide-by-slide presentation notes. |

## 一份公告与一个产品系列

同一官方编号的不同款式合为一行。一个产品系列收到多份官方公告时，每份公告各占一行。做五折或留出核对时，同名公告及从原文核实相连的公告一起分配。不同名称的同型号仍可能遗漏，同名只供筛查。Brompton 电助力折叠自行车的两份公告说明为什么要保留各自的日期和失效原因。 [Brompton 21-116](https://www.cpsc.gov/Recalls/2021/Brompton-Bicycle-Recalls-Electric-Folding-Bicycles-Due-to-Fall-and-Injury-Hazards) · [Brompton 22-155](https://www.cpsc.gov/Recalls/2022/Brompton-Bicycle-Recalls-Foldable-Electric-Bicycles-Due-to-Crash-and-Injury-Hazards)

## 三种不同的用法

- **找规律。** 全量 Apriori 先列每类常见的危险和分母。卡方可检查大类与危险整体上是否有关。聚类先按文字相似性分组，再用保留的公告检查各组危险构成。
- **检验能猜中多少。** 五折让每份公告都有一次折外预测。比较时逐步加入设计阶段可用的信息。模型的词表、PCA、参数和规则按各折的训练与内部验证数据拟合。召回原因用于理解失效和核对标签。
- **复核规律并回查部件。** 四折发现的规则用第五折的真实标注核对，模型提示作为辅助比较。原公告确认失效部位。比例反映已收录公告，估计设计本身的事故概率还需要销量、使用暴露和事故数据。

## 数据接口与复现

新版的 01、02 Notebook 都调用 `project_data.load_notices()`。它读取同一份公告级数据，每个 CPSC 公开编号一行。01 还用 `load_notice_audit()` 查看匹配证据。直接复跑 01、02 可以使用仓库已有的修订公告和审计文件。

| 保留文件 | 用途 |
|---|---|
| [data/dataset_corrected.json](data/dataset_corrected.json) | 两本 Notebook 的共同输入，一份官方公告一行。 |
| [data/reconciliation_summary.json](data/reconciliation_summary.json) | 清洗前后计数、数据来源和哈希。 |
| [data/source_row_audit.jsonl](data/source_row_audit.jsonl) | 逐来源行的官方编号映射、未解决记录和核对证据。按行存储，便于逐条查询。 |
| [scripts/project_data.py](scripts/project_data.py) | 准备项目数据并提供 Notebook 共用的读取接口。 |
| [scripts/source_preprocessing.py](scripts/source_preprocessing.py) · [scripts/reconcile.py](scripts/reconcile.py) | 统一入口调用的来源行清洗与官方公告核对。 |
| [scripts/product_features.py](scripts/product_features.py) | 两本 Notebook 共用的产品文字清理、向量化与公告分组。 |
| [scripts/association_rules.py](scripts/association_rules.py) | 全量发现与折内核对共用的 Apriori 规则统计。 |

复跑分析需要 NumPy、pandas、SciPy、Matplotlib、scikit-learn、PyTorch、transformers、mlxtend 和 XGBoost。使用本地 MiniLM 权重时设置 `RECALL_MINILM_PATH`。

从源头复建时，准备原始 RECALL-MM `dataset.json`、CPSC 历史 API JSON 和 CPSC 公告清单 CSV。运行 [DataPreprocessing.ipynb](../DataPreprocessing.ipynb) 前，在第一个代码单元填好三个路径，再执行 **Run All**。同一入口读取原始 JSON 一次，用来源行视图做质量检查，同时用原始行号和文字证据与官方快照匹配。快照缺失时，Notebook 生成来源行视图，并在第 4 节说明当前显示的是已有公告快照还是尚待生成。输入来源和哈希见核对汇总。

从仓库根目录运行下面的命令也会调用同一个准备函数。

```sh
python recall_audit/scripts/project_data.py dataset.json cpsc_api.json cpsc_listing.csv
```

复建同时生成 `preprocessing_output/` 中的原始行清洗表、质量清单和早期实验交接表。质量清单中的 `source_row` 与公告核对日志的来源行号相同，可以从修订公告的 `source_rows` 回查。

原 A/B/D 通过 `load_source()` 读取同版原始来源行，原 C 通过 `load_legacy_rows()` 读取交接表。这样原实验继续以原编号和大类为单位，新分类与聚类以官方公告为单位。
