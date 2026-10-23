# v2.3（B）stagger 交错耦合判据研究

基线：main@7f57cc7（物理代码未改；所有数字均为仿真输出，不称实测）。

- `V23_SCOPE.md`：研究范围（管家评审有条件通过，R1/R2 已修订）。
- `LIGHT_RESULTS.md`：轻算例冒烟结果（9/9 组已跑完，2026-10-23）。
- `data/stagger_light_matrix.json`：轻算例完整数据。
- `run_stagger_matrix_light.py`：轻算例脚本（S4 级，t_end=0.1，1 个 EM 区间，
  iter ∈ {1,2,3} × tol ∈ {None,1e-2,1e-3}）。
- `run_stagger_matrix_cloud.py`：云端全矩阵脚本（A1 级，t_end=0.4，4 个 EM
  区间，iter ∈ {1,2,3,5} × tol ∈ {None,1e-2,1e-3,1e-4 K} = 16 组）。

## 云端运行（请管家在 InstaBox 执行）

```bash
git checkout v2.3-stagger-study
python3 studies/v23_stagger/run_stagger_matrix_cloud.py
```

输出：`studies/v23_stagger/data/stagger_cloud_matrix.json`
（16 组完整记录 + 逐区间跨跑收缩因子）。

预计耗时：轻算例 iter=3/1 区间约 300 s；A1 级 4 区间、iter=5 档单组约
1 小时以内；16 组总计约 4–8 小时。可按 iter 档分批跑：
`_ITERS` 在脚本内可调（每档结果独立，JSON 分档追加由管家按需处理）。
