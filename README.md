# ABI-AKI 在线风险计算器（P2 部署件）

## 运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

界面：英文默认，侧栏可切中文（bilingual）。

## 组成
- `app.py` — Streamlit 双页签：① 单点计算器（训练集中位画像底座 defaults.json + 12 核心特征调节）② 批量 CSV 预测（管线特征行 → 逐检查点校准风险）
- `defaults.json` — 训练集（M4, 2008-2017）291 特征中位画像（典型患者底座）
- 模型工件：`model/n6_abi_gbm_frozen.joblib`（291 特征 LightGBM, D33；11.3 MB）+ `model/g2_recalibration_dca.json`（select-fit 重校准层系数，活读不硬编码）。仓库即云部署包；本地开发若 model/ 缺失自动回退 `../results/`

## 部署链（skill 12.2 checklist）
- [x] 模型序列化 + 版本（2026-09-22, D33/D34 链）
- [x] 校准链嵌入（原始概率→select-fit 校准概率；分带 0.10/0.20）
- [x] 临床语言输入 + 合理范围
- [x] 风险分层色带 + 行动建议语句（演示性）
- [x] 免责声明（研究用途，未前瞻验证）
- [x] 本地运行隐私（数据不上传）
- [ ] Streamlit Cloud URL（待 PI 部署后回填手稿）

## 验证状态
- 链路等价性：predict_risk 与冻结模型直算+g2 层 3 行逐值一致（2026-09-22 冒烟）
- 敏感性：典型患者 4.5% → 少尿+净排出 12.9% → +Cr↑/升压药 14.2%（方向幅度合理）
- UI 实测：渲染/分带/行动建议/免责全过（Playwright，2026-09-22）
