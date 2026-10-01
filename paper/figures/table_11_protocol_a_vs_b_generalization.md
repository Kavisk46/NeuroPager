Table 11 -- The central generalization comparison (Section 18): learned policy's paired advantage over LRU in page faults (negative mean_diff = learned had FEWER faults, i.e. better), Protocol A (within-distribution) vs Protocol B (leave-one-family-out), per family and horizon. Source: paired_comparisons_vs_learned['Learned_vs_LRU'] in each checkpoint, read verbatim (mean_diff, ci_95_low/high, paired_ttest_pvalue) -- these are the exact bootstrap/paired-test outputs scripts/generalization_experiment_2.py already computed, not recomputed here.

| Family | H | A: Learned-LRU mean diff | A: 95% CI | A: p (paired t) | B: Learned-LRU mean diff | B: 95% CI | B: p (paired t) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_uniform_random | 25 | 1.70 | [-4.20, 7.60] | 0.56 | 0.26 | [-2.44, 2.95] | 0.852 |
| A_uniform_random | 50 | 3.97 | [-0.82, 8.75] | 0.101 | 1.07 | [-1.79, 3.93] | 0.462 |
| A_uniform_random | 100 | 1.03 | [-5.96, 8.03] | 0.765 | 1.01 | [-2.08, 4.10] | 0.52 |
| A_uniform_random | 200 | 3.30 | [-4.05, 10.65] | 0.366 | 0.42 | [-2.54, 3.38] | 0.78 |
| A_uniform_random | 500 | -4.17 | [-14.49, 6.16] | 0.416 | -0.21 | [-3.02, 2.59] | 0.88 |
| B_temporal_locality | 25 | 0.60 | [-0.03, 1.23] | 0.0621 | 5.87 | [5.17, 6.57] | 6.2e-39 |
| B_temporal_locality | 50 | 0.03 | [-0.15, 0.22] | 0.712 | 6.62 | [5.87, 7.37] | 1.53e-41 |
| B_temporal_locality | 100 | 0.43 | [-0.09, 0.96] | 0.102 | 5.58 | [4.82, 6.35] | 7.72e-33 |
| B_temporal_locality | 200 | 0.10 | [-0.13, 0.33] | 0.375 | 4.88 | [4.27, 5.49] | 1.04e-36 |
| B_temporal_locality | 500 | 0.07 | [-0.13, 0.26] | 0.489 | 6.01 | [5.29, 6.74] | 5.6e-39 |
| C_bursty | 25 | 1.77 | [-0.62, 4.16] | 0.141 | -0.57 | [-1.00, -0.15] | 0.00809 |
| C_bursty | 50 | 0.47 | [-1.37, 2.30] | 0.607 | -0.34 | [-0.72, 0.04] | 0.0809 |
| C_bursty | 100 | 0.30 | [-1.27, 1.87] | 0.698 | -0.20 | [-0.57, 0.18] | 0.308 |
| C_bursty | 200 | 1.20 | [-0.30, 2.70] | 0.114 | -0.35 | [-0.79, 0.08] | 0.109 |
| C_bursty | 500 | 0.27 | [-1.29, 1.82] | 0.729 | -0.19 | [-0.54, 0.16] | 0.282 |
| D_long_range_reuse [structural-null] | 25 | n/a | n/a | n/a | 0.00 | [0.00, 0.00] | nan |
| D_long_range_reuse [structural-null] | 50 | n/a | n/a | n/a | 0.00 | [0.00, 0.00] | nan |
| D_long_range_reuse [structural-null] | 100 | n/a | n/a | n/a | 0.00 | [0.00, 0.00] | nan |
| D_long_range_reuse [redundant w/ H=500] | 200 | -30.00 | [-30.00, -30.00] | 0 | -5.00 | [-5.00, -5.00] | 0 |
| D_long_range_reuse [redundant w/ H=200] | 500 | -30.00 | [-30.00, -30.00] | 0 | -3.00 | [-3.00, -3.00] | 0 |
| E_phase_changing | 25 | -5.73 | [-14.97, 3.50] | 0.214 | 72.17 | [69.31, 75.04] | 4.23e-114 |
| E_phase_changing | 50 | -7.87 | [-17.31, 1.57] | 0.099 | 75.15 | [72.12, 78.18] | 6.49e-113 |
| E_phase_changing | 100 | -8.17 | [-17.45, 1.11] | 0.0823 | 76.67 | [74.11, 79.24] | 4.79e-128 |
| E_phase_changing [redundant w/ H=500] | 200 | -2.93 | [-10.26, 4.39] | 0.42 | 23.41 | [21.49, 25.33] | 1.24e-60 |
| E_phase_changing [redundant w/ H=200] | 500 | -3.43 | [-10.82, 3.95] | 0.35 | 74.42 | [72.41, 76.44] | 2.35e-145 |
| F_non_stationary | 25 | -153.30 | [-159.43, -147.17] | 5.57e-30 | 101.23 | [98.65, 103.81] | 2.35e-150 |
| F_non_stationary | 50 | -150.30 | [-156.72, -143.88] | 3.75e-29 | 118.50 | [116.32, 120.69] | 7.28e-178 |
| F_non_stationary | 100 | -138.93 | [-145.61, -132.25] | 1.1e-27 | 79.75 | [77.39, 82.10] | 4.6e-138 |
| F_non_stationary | 200 | -113.47 | [-120.53, -106.41] | 1.67e-24 | 82.44 | [80.32, 84.57] | 2.06e-149 |
| F_non_stationary | 500 | -21.10 | [-31.24, -10.96] | 0.0002 | 255.81 | [251.53, 260.09] | 4.51e-186 |
