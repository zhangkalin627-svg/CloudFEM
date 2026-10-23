# v2.3（B）stagger 交错耦合判据研究

基线：main@7f57cc7（物理代码未改；所有数字均为仿真输出，不称实测）。

- `V23_SCOPE.md`：研究范围（管家评审有条件通过，R1/R2 已修订）。
- `LIGHT_RESULTS.md`：轻算例冒烟结果（9/9 组已跑完，2026-10-23）。
- `data/stagger_light_matrix.json`：轻算例完整数据。
- `run_stagger_matrix_light.py`：轻算例脚本（S4 级，t_end=0.1，1 个 EM 区间，
  iter ∈ {1,2,3} × tol ∈ {None,1e-2,1e-3}）。
- `run_stagger_matrix_cloud.py`：云端全矩阵脚本（A1 级，t_end=0.4，4 个 EM
  区间，共 19 组：收缩档 iter∈{1,2,3,4,5}×tol=None（5 组，iter=4 仅取 d4）
  + tol 扫掠档 iter∈{1,2,3,5}×tol∈{1e-2,1e-3,1e-4} K（12 组）
  + 假设档 iter=5×tol∈{0.1,1.0} K（2 组，0.1–1K 预注册假设验证档））。
  判据见 `CLOUD_CRITERIA.md`（管家复核修订版）。

## 云端运行（请管家在 InstaBox 执行）

```bash
git checkout v2.3-stagger-study
python3 studies/v23_stagger/run_stagger_matrix_cloud.py
```

输出：`studies/v23_stagger/data/stagger_cloud_matrix.json`
（19 组完整记录 + 逐区间相邻档收缩因子）。

预计耗时：轻算例 iter=5/1 区间约 313 s；A1 级 4 区间、iter=5 档单组约
1 小时以内；19 组总计约 5–9 小时。可按档分批跑：
`_CONFIGS` 在脚本内可调（每组结果独立，JSON 分档追加由管家按需处理）。
