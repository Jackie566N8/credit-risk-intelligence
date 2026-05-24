PAGE_STEPS = {
    "Portfolio Overview": [
        "确认顶部状态为 Ready for demo，说明结果文件和模型 artifact 已加载。",
        "先查看 Project Report Highlight，说明报告中的随机森林核心指标。",
        "再查看 Reproducible Training Results，对比当前可复现实验指标。",
        "切换 Metric 下拉框，观察不同模型在 ROC-AUC、Recall、F2 等指标上的差异。",
    ],
    "Model Detail": [
        "在 Model 下拉框选择一个模型。",
        "查看 Cross Validation，判断模型稳定性。",
        "查看 Holdout Test，对比优化阈值和默认 0.50 阈值。",
        "查看 Feature Importance，解释主要风险因素。",
    ],
    "Single Customer Decision": [
        "填写贷款金额、收入、FICO、DTI、信用利用率、利率和贷款期限。",
        "确认贷款用途、住房状态、收入验证状态和 LGD 参数。",
        "点击 Run Assessment，等待模型预测和审批规则计算。",
        "查看审批建议、信用评分、预期损失和主要风险驱动因素。",
    ],
    "Batch Risk Assessment": [
        "上传包含必需字段的 CSV；没有文件时系统会使用样例客户。",
        "检查 Required columns，确认字段名称一致。",
        "等待批量评分完成，查看审批分布和信用分/违约概率散点图。",
        "下载评估结果 CSV 或模板 CSV。",
    ],
    "EDA Figures": [
        "确认已运行 `python modeling/plot.py` 生成图表。",
        "先查看 Accepted Loans 图表，解释违约样本、FICO、DTI、信用等级等特征。",
        "再查看 Rejected Applications 图表，补充申请端风险画像。",
        "将图表结论与模型特征重要性连接起来说明业务逻辑。",
    ],
}
