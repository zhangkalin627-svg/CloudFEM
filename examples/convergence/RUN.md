# v2.0 时间收敛证据 · 云端运行说明

分支 `v2.0-conv-cloud`（基线 5b59b4c，即管家已验收 27/27 的 v2.0-thermal-coupling）。

## 跑什么

`examples/convergence/` 下 6 个 case（7-case 队列除 C3 外全部；C3=dt0.005×em_every1 最细档在缪斯本机跑）：

| case | dt (s) | em_every |
|------|--------|----------|
| A1 | 0.02 | 5 |
| A2 | 0.01 | 5 |
| A3 | 0.005 | 5 |
| B1 | 0.02 | 1 |
| C1 | 0.01 | 1 |
| B2 | 0.02 | 2 |

物理设置（与 tests/test_tet10_em_thermal_coupled.py::_coupled_run 一致）：
铁圆柱 R=15mm L=60mm 置于空气盒（half=0.045, h=0.016），
B_applied=MU0×1e6 A/m（z 向），f=2kHz，t_end=0.4s，
绝热，T0=20°C，em_tol=1e-3，em_max_iter=60，em_relax=0.5，
热步 BE，stagger_iter=1（管家已确认）。

## 怎么跑

环境：python3.13，numpy 2.5.4，scipy（仓库仅依赖 numpy/scipy）。

```bash
git clone -b v2.0-conv-cloud https://github.com/zhangkalin627-svg/CloudFEM.git
cd CloudFEM
# 单 case（从仓库根目录，PYTHONPATH=.）：
PYTHONPATH=. python3 examples/convergence/dt_em_conv.py \
  --dt 0.02 --em-every 5 --case A1 --outdir ./conv_results
# 或按 cases.json 循环（各 case 独立进程，可并行）：
PYTHONPATH=. python3 - <<'EOF'
import json, subprocess
cases = json.load(open('examples/convergence/cases.json'))['cases']
for c in cases:
    subprocess.run(['python3', 'examples/convergence/dt_em_conv.py',
                    '--dt', str(c['dt']), '--em-every', str(c['em_every']),
                    '--case', c['case'], '--outdir', './conv_results'],
                    check=False)
EOF
```

说明：
- 每个 case 独立进程；同一 case 重复跑会跳过（`conv_results/<case>.json` 已存在则 SKIP）。
- `--outdir` 请显式指定（脚本默认 outdir 是缪斯本机绝对路径）。
- 本机参考耗时（2 核 VM，schema v2）：A1≈1300s、A2≈1100s；C3 最细档显著更久。

## 交回什么

`conv_results/` 下 6 个 JSON（A1/A2/A3/B1/C1/B2），原样发回缪斯整理偏差表。
JSON 字段（schema v2，均为仿真输出）：Tmax_end_C、E_in_J、mu_r_surf_mean_last、
P_last_W、P_total_last_W、**T_surf_peak_C**（表层峰值）、**t_cross_Tc_s**（穿越居里点时间，
"未穿越"表示 t_end 内未穿越）、em_iterations、P_first_W、balance_rel、wall_s、rc。

## 判据

各 case 的 T_surf_peak_C、t_cross_Tc_s、P_total_last_W 相对最细档 C3
（dt=0.005×em_every=1，本机在跑，结果随后补）偏差 < 2%。
