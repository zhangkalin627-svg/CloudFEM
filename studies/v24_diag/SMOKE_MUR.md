# v2.4 第一阶段 —— μr 可得性冒烟验证结论（SMOKE_MUR）

> 对应 scope §7 风险1 与管家评审"μr 内部值可得性只是待核提议，冒烟先核"。
> 验证基线：main@7f57cc7（分支 v2.4-diag，物理代码未改；
> 本脚本 `studies/v24_diag/run_mur_smoke.py` 只读驱动返回）。
> S4 级轻算例（t_end=0.1, dt=0.02, em_every=5，1 区间；
> 铁柱 R=15mm/L=60mm，气盒 half=0.045，h=0.016，f=2kHz，
> B_applied H0=1e6 A/m，T0=20°C，cp(T) 关闭，绝热）。
> 本机 numpy 1.26.4（定性冒烟；验收以管家云端 numpy 2.5.4 为准）。
> 全部数值为仿真输出；wall 时间为本机实测。
> 数据：`studies/v24_diag/mur_smoke.json`（2026-10-24）。

## 结论：A) μr 可得 —— 无需改 em3d_tet10.py

驱动侧 history 已现成携带割线 μr，无需扩展 EM 出参：

1. `history[iv]["mu_r_iron"]` 存在（ndarray，122 个工件单元，
   与 iron_idx / sub.element_volumes() 同序同长）；
   同行的 `mu_r_iron_mean/min`、`frac_above_Tc`、`Q_iron` 均存在。
2. **逐位相等**：history 的 `mu_r_iron` 与末 pass EM 解的
   `em_info["mu_r_elem"][iron_idx]` 最大绝对差 = 0.0，
   即驱动侧拿到的就是该 pass 求解内部用的割线 μr，
   不存在"驱动侧拿不到"的问题。
3. 聚合量可直接由驱动侧算出（本算例实测）：
   min=3.252 / 体积加权均值=101.716 / max=200.625，
   序关系成立；计数均值 95.762 ≠ 体积加权均值 101.716，
   说明体积加权口径必须写死（已入 CLOUD_CRITERIA §0）。
4. `frac_above_Tc` 现有字段确为**单元计数分数**
   （与 np.mean(Te > t_curie) 逐位一致）；体积分数可直接算出
   （本算例 Tmax=611.09°C < Tc=770.0°C，两者皆为 0.0，如预期）。

## 取样点不同态的实证（支撑 CLOUD_CRITERIA §0 不同态声明）

iter=1（μr 取自 T0 输入场）vs iter=3（μr 取自末 pass 输入场）：
体积加权均值 107.545 vs 101.716，**两者不等** ——
实证 μr_agg 绑定的是"该 pass EM 解的输入温度场"，
而 frac_above_Tc 取自"末 pass 输出的区间末温度场"；
两者时间相邻但不同态，故 v2.4 诊断只许并列记录，
不作时序因果断言（评审条件⑦后半）。

## 对实施的含义

- 不需要改 `em3d_tet10.py`（结论 A，无需报核准清单）；
- v2.4 驱动改动只需在 induction3d_tet10.py 追加记录：
  `stagger_dT_seq`、体积分数版 `frac_above_Tc`、
  体积加权版 `mu_r_agg` 三元组（取样点按 CLOUD_CRITERIA §0）；
- 本机 wall：iter=3 约 321 s，iter=1 约 227 s（本机实测，
  仅供量级参考，非验收数字）。
