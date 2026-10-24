# v2.4 诊断补强 —— D6 交卷说明

> 分支：v2.4-diag；基线：main@7f57cc7；准则：studies/v24_diag/CLOUD_CRITERIA.md
> （终稿冻结于 4f614f9，F1–F3 + D5 措辞终稿）。
> 本文件为交卷底稿：本地结果只作冒烟，全部数字为**仿真输出**；
> 验收数字以管家云端新鲜环境（numpy 2.5.4 / python3.13）复跑为准。

## 1. 本次提交

- 提交号：代码与验证提交见下（本文档提交随后推送；远端以
  `git ls-remote origin v2.4-diag` 复核值为准，交卷信附实测 hash）
- 内容：
  - `cloudfem/physics/induction3d_tet10.py`：驱动新增默认关闭的诊断
    开关 `diag`（仅追加记录，不动任何求解数值路径；签名仅在末尾
    新增 `diag=False`，stagger_iter / stagger_tol / em_every 默认值
    未改；em3d_tet10.py 未动）。
  - 诊断记录（每 EM 区间一条，取样点按准则 §0）：
    - `stagger_dT_seq`：逐 pass dT 序列，长度 = 实际 passes；
      首 pass 记 None（显式缺失，绝不 0 填充）；标量 `stagger_dT`
      保留 = 末 pass 值（现状语义不变）。
    - `frac_above_Tc_vol`：区间末温度场下工件单元超居里点
      **体积分数**（独立新键）；旧键 `frac_above_Tc`（单元计数
      分数）语义与取值保持不变，两键不混用。
    - `mu_r_agg`：该区间末 pass EM 解所用割线 μr 三元组
      (min, 体积加权均值, max)，取自末 pass **输入**温度场；
      与 frac_above_Tc_vol（末 pass **输出**场）不同态，只并列
      记录，不作时序因果断言。
  - `tests/test_v24_diag.py`：新增测试（诊断关无诊断键 / D2 结构 /
    早停结构 / D1 数值复现小算例）。
  - `studies/v24_diag/run_diag_verify.py` + `_run_s4_case.py`：
    D1/D2 本地冒烟验证脚本；结果 `studies/v24_diag/diag_verify.json`。
  - 本文件。

## 2. 云端验收矩阵（D6 写死 7 组，不得扩大）

S4/A1 级算例口径沿 v2.3 轻算例与本研究脚本
（studies/v24_diag/_run_s4_case.py 的算例定义可作参照；
云端以管家复核的 A1 级算例为准）：

| 组 | 诊断 | stagger_iter | stagger_tol |
|---|---|---|---|
| 1–5 | 开 | 1, 2, 3, 4, 5 | None |
| 6 | 关（对照） | 1 | None |
| 7 | 关（对照） | 5 | None |

判定：D1 数值复现（容差 ≤1e-12 量化版，硬判据；诊断开/关与基线
的 Tmax / E_in / history 既有浮点键相对差 ≤1e-12，整数/布尔键
精确相等，balance_rel ≤4.4e-16 量级）与 D5 全回归共同生效；
D2 结构（seq 长度=passes、frac_above_Tc_vol∈[0,1]、三元组有序、
首 pass 缺失标记）；D3 按缺失规则（iter≥3 全可算对照 v2.3 跨运行
比值，iter=1/2 标“序列不足”，d(s)=0 记缺失不取 inf）；
D4 只许三态措辞（数据支持 / 数据不支持 / 数据不足以判定）。

## 3. 云端一键运行入口（建议）

```sh
git checkout v2.4-diag   # 应到 __COMMIT__
# 新增测试 + 全回归（逐文件运行，rc=0 为过）
for f in tests/test_*.py; do python3 "$f" || echo "FAIL $f"; done
# D1/D2 冒烟验证（S4 级，含 main@7f57cc7 临时基线 worktree，自动清理）
python3 studies/v24_diag/run_diag_verify.py
```

## 4. 本地冒烟结果摘要（仿真输出；本机 numpy 1.26.4 / python 3.12.3）

- D1(a) 诊断关 vs main@7f57cc7 基线（iter 1–5）：全部通过。
  Tmax / E_in / history 既有浮点键（含旧键 frac_above_Tc）相对差
  **0.0**（逐位一致），整数/布尔键精确相等，mu_r_iron / Q_iron
  数组逐位相等，T_hist 逐位相等。各 iter Tmax / E_in（仿真输出）：
  iter=1：666.6674 °C / 46898.189900 J；iter=2：611.0685 / 47135.511377；
  iter=3：611.0876 / 47139.723677；iter=4：609.5927 / 47104.096045；
  iter=5：609.1611 / 47064.721296。
- D1(b) 诊断开 vs 诊断关（iter 1–5）：全部通过，相对差同样 **0.0**。
- D1 补充说明（balance_rel 量级，**如实标注，提请复核注意**）：
  iter=3、4 的 |balance_rel| 分别为 6.17e-16、7.72e-16，略高于
  4.4e-16 参考量级；但两值在 v2.4 与 main@7f57cc7 基线两侧
  **逐位相同**（相对差 0.0），即诊断未使能量对账恶化分毫——
  该量级是基线求解器本身在多 pass 下的累积舍入（同为 1e-16
  量级），非 v2.4 回归。验证脚本的 ok 判定按“不劣于基线”
  口径通过；若按“每组严格 ≤4.4e-16”解读，则基线自身亦不
  通过，提请管家在 D6 复核时裁定口径。
- D2 结构（iter 1–5 + 早停组）：全部通过。逐 pass 序列
  （仿真输出）：iter=1：[None]；iter=2：[None, 68.711]；
  iter=3：[None, 68.711, 13.131]；iter=4：[None, 68.711, 13.131, 2.558]；
  iter=5：[None, 68.711, 13.131, 2.558, 1.232]；早停组（iter=5,
  tol=1e6）：[None, 68.711]，passes=2（序列长度=实际 passes）。
  各组 frac_above_Tc_vol = 0.0（本 S4 算例区间末温度未超居里点），
  mu_r_agg 三元组有序（如 iter=1：3.319 / 107.545 / 200.588；
  iter=3：3.252 / 101.716 / 200.625），与第一阶段冒烟值一致。
- D2 验证明细：studies/v24_diag/diag_verify.json（D1/D2 全量，
  all_ok=true，总耗时 6158.7s 本机实测）。
- D5 全回归：既有 **28/28** 个测试文件全过（逐文件 rc=0，
  总耗时 3534.4s 本机实测；最慢为 test_tet10_em.py 1119.5s、
  test_tet10_em_thermal_coupled.py 638.4s、test_tet10_cp_enthalpy.py
  561.0s），新增 tests/test_v24_diag.py 全过（诊断关无诊断键 /
  D2 结构 / 早停结构 / D1 数值复现小算例，max rel diff 0.0）。
  本地环境为 numpy 1.26.4 / python 3.12.3（与云端 2.5.4 /
  python3.13 不同属预期；本地只作冒烟，验收以云端复跑为准）。
- D5 逐文件日志备查：本地运行记录（v24_scratch/d5_logs，不入库）。

## 5. 纪律声明

- 未合 main；未动 v2.2-robin-cooling；未改准则文件；未扩大矩阵。
- 诊断开启只增记录：验证中 T_hist 与既有 history 键逐位/容差内
  一致（见 diag_verify.json），数值路径未动。
- 未决点②的判定措辞严格按 D4 三态，云端出数后于研究报告中给出。
