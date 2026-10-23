# CloudFEM v2.2 开发报告（A：Robin 对流冷却边界）

分支：v2.2-robin-cooling（候选，不合 main）。基线 main@7f57cc7。
范围依据：docs/v2.2_scope.md（管家评审 R1–R5 修订版）。全部数字为
"仿真输出/仿真结果"；h = 5000 W/(m²·K)、T_inf = 20 °C 为模型假设
示例值、非实测数据。

## 1. Scope 对照表

| scope 条款 | 落实 |
|---|---|
| §2.1 冷却模型 q_loss = h·(T_s − T_inf)，恒定等效 h、全部外表面 | `cooling={"surface": (h, T_inf)}` 参数；h/T_inf float 或 callable(x,y,z)；h<0 → ValueError；未知键 → ValueError；h=0 逐位退回绝热 |
| §2.2 子网格外表面登记（面发生率计数，禁用节点集推面） | `_register_surface_boundary_tet10`（新 helper）；`_cooling_boundary_terms` 复用 v1.6 `assemble_boundary_terms_tet10`；S2 守门通过（含薄壁陷阱负对照） |
| §2.3 步进改造：A_sys = C + θ·h·(K_th+K_h)，RHS 每步 +h·F_h | `_make_solver` 内实现；K_h/F_h 一次装配（h 不随时间变）；cp(T) 区间重建兼容 |
| §2.4 三方对账 E_in = ΔH_disc + E_loss；balance_rel 新口径 | 已实现；E_loss 逐热步 theta 加权累加；绝热时 E_loss 恰为 0.0，balance_rel 逐位沿 v2.0/v2.1 |
| R2：balance_cool = (E_in − ΔH − E_loss)/D，D = max(\|ΔH\|, E_loss) | 已实现；D=0 返回 None（记"不适用"，不许记为通过） |
| R3：peak_engaged 诊断（每热步末单元温度落入 [Tc−w, Tc+w]） | history 每区间新增 `peak_engaged`；判据形式固定 |
| §3 逐位一致构造保证（cooling=None 跳过 K_h/F_h 装配） | `K_h is None` 分支；S1 双查 |
| R1：§1.1 降格 + §7 已知限制 | docs/v2.2_scope.md 与 README v2.2 条目逐字携带 |
| R4：阈值纪律 | 判据形式先行写死；数值阈值本地冒烟后写死进测试（见 §3）；出数后调整重新送审 |
| R5：S3 的 Q≡0 构造开工定死 | J_s 零激励 callable（测试注释写死，不许事后更换） |
| §5 不做事项 | B/C/时变 h/非均匀 T0/Neumann/Dirichlet/k_th(T) 均未动；em3d_tet10.py 一行未改；em_every=5 默认未动 |

## 2. 判据形式清单（§4.1 冒烟，先行写死）

- S1：绝热退化逐位回归（跨机安全设计，见 §8 修订）。(a) cooling=None 与 h=0 两路
  逐位一致（max |dT| == 0.0，E_in/balance_rel 相等）；E_loss 恰为 0.0。
  (b) 同机即时独立参照：驱动（J_s 零激励，F_Q ≡ 0，T0=100 °C）vs
  transient_heat_tet10（f=0、无 Robin）同子网格，终场最大相对差 ≤ 1e-12。
  阈值未放宽。
- S2：Robin 装配守门。驱动 K_h/F_h vs 独立面发生率枚举 + 同被积函数装配，
  相对差 ≤ 1e-12；铁柱子网格（100 外表面）+ 一元件厚薄壁（32 面）+ 双锥体
  陷阱几何（驱动 6 面 vs 禁用推法 7 面，负对照成立）；无 occurrence-2 面泄漏。
- S3：纯冷却交叉路径。驱动（J_s 零激励，F_Q ≡ 0）vs transient_heat_tet10
 （f=0、同 Robin、同子网格、同 T0），节点温度相对差 ≤ 1e-9；balance_cool
  在求解器精度量级。
- S4：三方对账。|balance_rel| ≤ 1e-9（加热冷却跑）；|balance_cool| ≤ 1e-9
  （纯冷却跑）；绝热 E_loss 恰为 0.0。
- S5：冷却方向性。Tmax(冷却) < Tmax(绝热) 严格；E_loss > 0；表面节点均温
  跨跑低于绝热路（surface-mean(cooled) < surface-mean(adiabatic)；见 §5 注）。

（R-2 修订 2026-10-23 独立复核：原"表面均温 < 芯部均温"字面不成立，
已按测试实际实现改为跨跑断言，与 §5 注一致。）

## 3. 阈值写死记录

| 判据 | 首跑实测值（仿真输出） | 写死阈值 | 裕度 |
|---|---|---|---|
| S1 退化+即时参照 | (a) h=0 vs cooling=None 逐位一致；(b) 同机独立参照终场最大相对差 | 1e-12 | 待新跑实测填入 |
| S2 dK_rel/dF_rel | 0.000e+00 / 0.000e+00（铁柱 100 面、薄壁 32 面、双锥体 6 面） | 1e-12 | 裕度无限（逐位） |
| S3 交叉路径相对差 | 0.000e+00 | 1e-9 | 裕度无限（逐位） |
| S3 balance_cool | −2.423e-15 | 1e-9 | ~6 个量级 |
| S4 balance_rel | 绝热 0.000e+00 / 冷却 −7.030e-17；E_loss 绝热恰 0.0 / 冷却 7.34e+02 J | 1e-9 | ~8 个量级 |
| S5 | Tmax 666.67→620.39 °C（严格下降）；E_loss>0；表面均温跨跑下降（见 §5 注） | 方向断言 | — |
| A4 balance_cool | 峰开 −1.001e-14 / 峰关 −3.450e-15（焓形式闭合，R-1 修订：两算例均为纯冷却、E_in≡0，驱动端 balance_rel 硬置 0.0，故断言改用 balance_cool；数字为实测非硬编码 0） | 1e-9 | ~5 个量级 |
| A4 peak_engaged | 25/30 区间为真（R3 门通过） | 方向断言（≥1 区间为真） | — |
| A4 t_cross | 峰开 0.32 s > 峰关 0.26 s（延迟方向与 v2.1 纯热延迟结论一致） | 方向断言（t_on > t_off） | 分辨率 0.02 s |
| A1 | Tmax 1499.60→1327.64 °C（严格下降）；E_loss=8363 J>0；表面均温跨跑 −164.88 °C | 方向断言 | — |
| A2 balance_rel | 9.973e-16 | 1e-9 | ~6 个量级 |

## 4. 冒烟结果

- S1（绝热退化逐位回归，修订后设计见 §8）：(a) cooling=None 与 h=0 两路逐位一致；
  E_loss 恰为 0.0；(b) 同机独立参照交叉验证。**待新跑验证**。
- S2（Robin 装配守门）：铁柱子网格（100 外表面）、一元件厚薄壁（32 面）、
  双锥体陷阱（驱动 6 面 vs 禁用推法 7 面，负对照成立），dK_rel/dF_rel
  均为 0.000e+00；无 occurrence-2 面泄漏。**通过**。
- S3（纯冷却交叉路径）：驱动（J_s 零激励，F_Q ≡ 0）vs transient_heat_tet10
 （f=0、同 Robin、同子网格），节点温度相对差 0.000e+00；
  balance_cool = −2.423e-15。**通过**。
- S4（三方对账）：绝热 balance_rel=0.0、E_loss 恰 0.0；冷却
  balance_rel=−7.03e-17、E_loss=734 J。**通过**。
- S5（冷却方向性）：Tmax 666.67→620.39 °C 严格下降；表面均温
  372.89→348.47 °C 跨跑下降；E_loss > 0。**通过**（口径见 §5 注）。
- A4（cp×冷却联合，本地 rc=0）：焓形式 balance_cool 峰开 −1.001e-14 /
  峰关 −3.450e-15（R-1 修订：纯冷却 E_in≡0、balance_rel 硬置 0.0，故断言改
  用 balance_cool，数字为实测非硬编码 0；阈值 1e-9，裕度 ~5 个量级）；
  peak_engaged 25/30 区间为真；t_cross 峰开 0.32 s > 峰关 0.26 s。**通过**。
- A1（0.4 s 全加热冷却跑，本地 rc=0）：Tmax 1499.60→1327.64 °C 严格下降；
  E_loss=8363 J；表面均温跨跑 −164.88 °C。**通过**。
- A2（三方对账全量）：balance_rel=9.973e-16。**通过**。
- A3（稳态平衡，t_end=60 s，本地 rc=0）：末段 P=8248 W vs q_loss=8227 W，
  相对差 2.5e-3；dTmax/dt=0.28 K/s。**通过**（阈值 1e-2 / 1.0 K/s）。
- v2.0/v2.1 回归（绝热路径）：本地跑 test_tet10_cp_enthalpy.py
  （ALL v2.1 cp(T) VALIDATIONS PASSED）与
  test_tet10_em_thermal_coupled.py
  （ALL THERMALLY COUPLED BH VALIDATIONS PASSED；关键数字与 scope 一致：
  表层 μr 6.05→1.00、功率比 0.313、EM iters [40,15,34,31]、
  对账 −4.0e-16 量级）。**无回归**。

## 5. 范围外观察

- `_FACES10` 的中点编号遵循 `EDGES = ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))`
  顺序；测试几何一律用库自带 `box_mesh_tet10` 构造，避免手写 enrichment
  顺序错位（已踩过一次，见开发记录）。
- Kuhn 网格（2×2×1、3×1×1）的内部面对角边中点恒在几何内部，节点集推法
  在此不咬陷阱；双锥体（两四面体共面）是最小陷阱几何，已用作 S2 负对照
  （驱动 6 面 vs 禁用推法 7 面）。
- **S5 口径说明**：scope §4.1 S5 原文"终点表面节点温度低于芯部"按字面在
  强趋肤加热工况下不成立——表面是焦耳热最集中的区域，有无冷却表面都是
  全场最热；冷却的效果是让表面比"不冷却时"更低。S5 改为跨跑断言
  （surface-mean(cooled) < surface-mean(adiabatic)），方向性不变；
  Tmax 严格下降与 E_loss > 0 均按原文通过。
- **A4 算例设计教训**：em_every=5 时区间冻结把焓峰跳过（v2.1 已知限制），
  峰开/峰关穿越时刻在 0.1 s 区间粒度下不可分辨；改 em_every=1（每步重建
  电容矩阵）后延迟 0.06 s 在 0.02 s 分辨率下清晰可辨。A4 固定用 em_every=1。
- L9 test-entry（run_all_tests.py）已并入本分支，随 v2.2 交付纳入云端验收。

## 6. 交云端清单（待交卷时填）

- tests/test_tet10_em_thermal_coupled_v22.py（S1–S5 冒烟 + A1–A5 全量断言）
- A1–A5 全量算例（本机只做 rc=0 可跑性确认，定量结论留云端）
- 28+1 全回归 rc=0（L9 双档入口随附）

## 8. 修订记录（验收发现③，2026-10-23）

- 原因：管家云端复核发现 S1 断言内嵌本机基线（Tmax=666.6673583840344 等），
  跨机浮点差异导致原阈值下失败——测试设计缺陷，不合 main。
- 改法：删除硬编码 `_BASE`；S1 改为 (a) h=0 vs cooling=None 逐位一致 +
  (b) 同机即时独立参照（transient_heat_tet10，f=0、无 Robin，同子网格、
  T0=100 °C 非平凡扩散工况）。阈值未放宽（s1_rel 仍为 1e-12）。
- 本节与 §2/§4 的 S1 条目已同步为新设计；旧基线数字不再作为断言依据。
