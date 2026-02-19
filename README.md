# 信贷风险智能分析与决策平台

基于 Python 构建的端到端信贷风险分析与决策平台，覆盖 LendingClub 数据清洗、特征工程、违约预测、信用评分、贷款审批建议、批量风险评估、预期损失计算和结果可视化。

## 项目功能

- 数据处理：使用 Pandas、NumPy 读取 LendingClub 2007-2018 年 accepted/rejected 公开贷款数据。
- 探索分析：使用 Matplotlib、Seaborn 输出目标分布、信用等级违约率、FICO 分箱、DTI、信用利用率、相关性热力图等图表。
- 模型训练：使用 Scikit-learn 对比逻辑回归、决策树、随机森林和梯度提升模型。
- 类别不平衡处理：使用 imbalanced-learn 的 SMOTE，并结合 5 折交叉验证和阈值优化。
- 决策输出：提供单客户违约概率、300-850 信用评分、审批建议和预期损失。
- 批量评估：支持上传 CSV 批量生成风险等级、审批建议和预期损失。
- Streamlit 平台：展示模型结果、单模型详情、EDA 图表、单客户决策和批量风险评估。

## 目录结构

```text
credit-risk-intelligence/
├── streamlit_app.py                 # Streamlit 应用入口
├── config/                          # 路径、页面、模型展示、审批阈值配置
├── pages/                           # Streamlit 页面模块
├── ui/                              # 通用 UI 组件
├── charts/                          # 可视化图表函数
├── risk/                            # 信用评分、违约概率、审批规则、批量评估
├── model_results/                   # 模型结果 txt 解析与读取
├── modeling/                        # 离线建模、EDA 绘图和训练产物
│   ├── train_credit_models.py       # 一次训练全部模型
│   ├── train_random_forest.py       # 单独训练随机森林
│   ├── plot.py                      # 生成 EDA 图表
│   ├── results/                     # 模型训练结果 txt
│   └── figures/                     # EDA 图片
├── data/                            # 原始数据目录，不纳入 git
├── pyproject.toml
└── README.md
```

## 数据文件

请将 LendingClub 数据放在以下位置：

```text
data/accepted/accepted_2007_to_2018Q4.csv
data/rejected/rejected_2007_to_2018Q4.csv
```

项目默认读取 accepted 数据中的贷款金额、年收入、FICO 区间、负债收入比、信用利用率、利率、期限、信用等级、用途等字段，并基于 `loan_status` 构造违约标签。

## 安装依赖

项目使用 `uv` 管理依赖：

```bash
uv sync
```

如果已存在 `.venv`，也可以直接使用：

```bash
.venv/bin/python -m py_compile streamlit_app.py config/*.py charts/*.py ui/*.py risk/*.py pages/*.py model_results/*.py modeling/*.py
```

## 生成 EDA 图表

```bash
.venv/bin/python modeling/plot.py
```

输出目录：

```text
modeling/figures/
```

## 数据探索与图表解读

### 违约样本分布

目标变量由 `loan_status` 构造，将 `Charged Off`、`Default`、`Late (31-120 days)` 等状态视为违约，将 `Fully Paid` 视为非违约。样本存在明显类别不平衡，因此后续模型训练使用 SMOTE 进行过采样。

![目标变量分布](modeling/figures/target_count.png)

### 信用等级与违约风险

信用等级越低，借款人的违约风险通常越高。该图用于展示 LendingClub 信用等级与违约率之间的关系，也为后续模型中特征重要性分析提供业务解释。

![信用等级违约率](modeling/figures/grade_default_rate.png)

### FICO 分数与违约风险

FICO 分数是信贷风险判断中的核心变量。项目将 `fico_range_low` 和 `fico_range_high` 取均值得到 `fico_score`，并按分数区间统计违约率，用于观察信用分下降时违约风险的变化。

![FICO 分箱违约率](modeling/figures/fico_band_default_rate.png)

### 贷款期限、信用等级与违约率

该热力图同时观察贷款期限和信用等级对违约率的影响。较长贷款期限通常带来更高不确定性，低信用等级叠加长期限时风险更集中。

![期限和信用等级违约率](modeling/figures/term_grade_default_rate.png)

### 收入、贷款金额与违约率

该图将年收入和贷款金额分箱后计算违约率，用于分析贷款金额相对于收入水平是否过高。它可以辅助解释平台中“贷款金额/收入比例”这一风险驱动因素。

![收入和贷款金额违约率](modeling/figures/income_loan_amount_default_heatmap.png)

### FICO、DTI 与贷款结果

散点图展示 FICO 分数、负债收入比和贷款结果之间的关系。低 FICO、高 DTI 的客户通常处于更高风险区域。

![FICO 和 DTI 散点图](modeling/figures/fico_dti_by_target.png)

### 信用利用率与贷款结果

信用利用率反映循环信用额度的使用压力。较高的 `revol_util` 往往意味着客户偿债压力较大，是风险评分中的重要参考变量。

![信用利用率分布](modeling/figures/revol_util_by_target.png)

### 数值特征相关性

相关性热力图用于检查核心数值特征之间的线性关系，例如贷款金额、利率、FICO、DTI、信用利用率和目标变量之间的关系。

![数值特征相关性](modeling/figures/numeric_correlation_heatmap.png)

### 被拒申请画像

rejected 数据用于观察被拒申请的风险分数、申请金额、DTI 和地区分布，辅助理解通过样本以外的申请端风险特征。

![被拒申请风险分数](modeling/figures/rejected_risk_score_distribution.png)

![被拒申请金额和 DTI](modeling/figures/rejected_amount_vs_dti.png)

## 训练模型

一次训练全部模型：

```bash
.venv/bin/python modeling/train_credit_models.py
```

单独训练某个模型：

```bash
.venv/bin/python modeling/train_random_forest.py
.venv/bin/python modeling/train_logistic_regression.py
.venv/bin/python modeling/train_decision_tree.py
.venv/bin/python modeling/train_gradient_boosting.py
```

默认使用分层抽样训练。全量训练可使用：

```bash
.venv/bin/python modeling/train_credit_models.py --sample-rows 0
```

训练结果输出到：

```text
modeling/results/
```

当前默认抽样训练结果摘要如下，完整结果见 `modeling/results/`：

| 模型 | CV ROC-AUC | Test ROC-AUC | 阈值 | Precision | Recall | F1 | F2 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7029 | 0.7013 | 0.3500 | 0.2674 | 0.8760 | 0.4098 | 0.6020 | 0.4642 |
| Decision Tree | 0.6580 | 0.6587 | 0.1200 | 0.2370 | 0.9419 | 0.3787 | 0.5906 | 0.3438 |
| Random Forest | 0.6985 | 0.6987 | 0.2800 | 0.2653 | 0.8815 | 0.4078 | 0.6019 | 0.4565 |
| Gradient Boosting | 0.7064 | 0.7060 | 0.1600 | 0.2714 | 0.8721 | 0.4140 | 0.6045 | 0.4757 |

默认结果中，梯度提升模型的 CV ROC-AUC 最高；如果更看重违约召回率，决策树在优化阈值后召回率最高。随机森林在召回率和业务可解释性之间较均衡，其特征重要性结果中利率、FICO、DTI、贷款期限、信用等级、信用利用率等变量具有较强解释价值。

## 启动 Streamlit 平台

```bash
.venv/bin/streamlit run streamlit_app.py
```

默认访问：

```text
http://127.0.0.1:8501
```

平台页面包括：

- Portfolio Overview：模型整体对比。
- Model Detail：单模型交叉验证、测试集结果、特征重要性和原始 txt。
- Single Customer Decision：单客户违约概率、信用评分、审批建议和预期损失。
- Batch Risk Assessment：批量 CSV 风险评估。
- EDA Figures：展示已生成的 EDA 图表。

## 平台页面说明

### Portfolio Overview

展示各模型的 ROC-AUC、Precision、Recall、F1、F2、Accuracy 等指标，便于比较不同模型在违约识别任务中的表现。

### Model Detail

展示单个模型的交叉验证结果、测试集结果、默认阈值对比、特征重要性和原始结果文件。该页面适合用于项目答辩中的模型解释部分。

### Single Customer Decision

输入单个客户的贷款金额、年收入、FICO、DTI、信用利用率、利率、期限、贷款用途等信息，输出：

- 违约概率
- 300-850 信用评分
- 风险等级
- 审批建议
- 预期损失
- 主要风险驱动因素

### Batch Risk Assessment

支持上传 CSV 批量评估客户风险，并导出带有违约概率、信用评分、风险等级、审批建议和预期损失的结果文件。

### EDA Figures

集中展示 `modeling/figures/` 下的全部探索性分析图表，方便从数据层面解释模型输入和业务结论。

## 当前说明

当前 Streamlit 单客户决策模块使用独立评分规则计算违约概率、信用评分和审批建议；离线训练模块用于模型对比和结果展示。后续可以进一步将训练好的随机森林或梯度提升模型保存为 `joblib` 文件，并接入平台进行真实模型推理。
