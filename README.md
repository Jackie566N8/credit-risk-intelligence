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

## 当前说明

当前 Streamlit 单客户决策模块使用独立评分规则计算违约概率、信用评分和审批建议；离线训练模块用于模型对比和结果展示。后续可以进一步将训练好的随机森林或梯度提升模型保存为 `joblib` 文件，并接入平台进行真实模型推理。
