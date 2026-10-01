Table 9 -- Protocol A (within-distribution) operational results: mean page faults (+/- population std across test episodes), mean hit ratio, and mean Belady gap, per family/horizon/policy. Source: classical_policy_faults, learned_policy_faults, and belady_gaps fields of each protocol_a_*.json checkpoint; means computed here, no underlying value recomputed. D_long_range_reuse @ H in {25,50,100} has no Learned row: both models' status is structural_null_single_class_training_data (scikit-learn cannot fit on single-class data), so no model was trained or replayed.

| Family | H | Policy | Mean faults | Std | Hit ratio | Belady gap |
| --- | --- | --- | --- | --- | --- | --- |
| A_uniform_random | 25 | Belady_MIN | 813.6 | 11.4 | 0.593 | n/a |
| A_uniform_random | 25 | FIFO | 1439.9 | 16.4 | 0.280 | 0.77 |
| A_uniform_random | 25 | LRU | 1439.3 | 16.9 | 0.280 | 0.77 |
| A_uniform_random | 25 | LFU | 1437.3 | 15.0 | 0.281 | 0.77 |
| A_uniform_random | 25 | Random | 1435.7 | 13.1 | 0.282 | 0.76 |
| A_uniform_random | 25 | Learned | 1441.0 | 16.5 | 0.280 | 0.77 |
| A_uniform_random | 50 | Belady_MIN | 813.6 | 11.4 | 0.593 | n/a |
| A_uniform_random | 50 | FIFO | 1439.9 | 16.4 | 0.280 | 0.77 |
| A_uniform_random | 50 | LRU | 1439.3 | 16.9 | 0.280 | 0.77 |
| A_uniform_random | 50 | LFU | 1437.3 | 15.0 | 0.281 | 0.77 |
| A_uniform_random | 50 | Random | 1435.7 | 13.1 | 0.282 | 0.76 |
| A_uniform_random | 50 | Learned | 1443.2 | 15.6 | 0.278 | 0.77 |
| A_uniform_random | 100 | Belady_MIN | 813.6 | 11.4 | 0.593 | n/a |
| A_uniform_random | 100 | FIFO | 1439.9 | 16.4 | 0.280 | 0.77 |
| A_uniform_random | 100 | LRU | 1439.3 | 16.9 | 0.280 | 0.77 |
| A_uniform_random | 100 | LFU | 1437.3 | 15.0 | 0.281 | 0.77 |
| A_uniform_random | 100 | Random | 1435.7 | 13.1 | 0.282 | 0.76 |
| A_uniform_random | 100 | Learned | 1440.3 | 18.1 | 0.280 | 0.77 |
| A_uniform_random | 200 | Belady_MIN | 813.6 | 11.4 | 0.593 | n/a |
| A_uniform_random | 200 | FIFO | 1439.9 | 16.4 | 0.280 | 0.77 |
| A_uniform_random | 200 | LRU | 1439.3 | 16.9 | 0.280 | 0.77 |
| A_uniform_random | 200 | LFU | 1437.3 | 15.0 | 0.281 | 0.77 |
| A_uniform_random | 200 | Random | 1435.7 | 13.1 | 0.282 | 0.76 |
| A_uniform_random | 200 | Learned | 1442.6 | 18.8 | 0.279 | 0.77 |
| A_uniform_random | 500 | Belady_MIN | 813.6 | 11.4 | 0.593 | n/a |
| A_uniform_random | 500 | FIFO | 1439.9 | 16.4 | 0.280 | 0.77 |
| A_uniform_random | 500 | LRU | 1439.3 | 16.9 | 0.280 | 0.77 |
| A_uniform_random | 500 | LFU | 1437.3 | 15.0 | 0.281 | 0.77 |
| A_uniform_random | 500 | Random | 1435.7 | 13.1 | 0.282 | 0.76 |
| A_uniform_random | 500 | Learned | 1435.1 | 23.2 | 0.282 | 0.76 |
| B_temporal_locality | 25 | Belady_MIN | 0.2 | 0.4 | 1.000 | n/a |
| B_temporal_locality | 25 | FIFO | 6.2 | 3.9 | 0.997 | 6.00 |
| B_temporal_locality | 25 | LRU | 1.6 | 1.9 | 0.999 | 1.47 |
| B_temporal_locality | 25 | LFU | 18.4 | 14.6 | 0.991 | 18.20 |
| B_temporal_locality | 25 | Random | 8.7 | 3.6 | 0.996 | 8.51 |
| B_temporal_locality | 25 | Learned | 2.2 | 2.6 | 0.999 | 2.07 |
| B_temporal_locality | 50 | Belady_MIN | 0.2 | 0.4 | 1.000 | n/a |
| B_temporal_locality | 50 | FIFO | 6.2 | 3.9 | 0.997 | 6.00 |
| B_temporal_locality | 50 | LRU | 1.6 | 1.9 | 0.999 | 1.47 |
| B_temporal_locality | 50 | LFU | 18.4 | 14.6 | 0.991 | 18.20 |
| B_temporal_locality | 50 | Random | 8.7 | 3.6 | 0.996 | 8.51 |
| B_temporal_locality | 50 | Learned | 1.7 | 1.9 | 0.999 | 1.50 |
| B_temporal_locality | 100 | Belady_MIN | 0.2 | 0.4 | 1.000 | n/a |
| B_temporal_locality | 100 | FIFO | 6.2 | 3.9 | 0.997 | 6.00 |
| B_temporal_locality | 100 | LRU | 1.6 | 1.9 | 0.999 | 1.47 |
| B_temporal_locality | 100 | LFU | 18.4 | 14.6 | 0.991 | 18.20 |
| B_temporal_locality | 100 | Random | 8.7 | 3.6 | 0.996 | 8.51 |
| B_temporal_locality | 100 | Learned | 2.1 | 1.9 | 0.999 | 1.90 |
| B_temporal_locality | 200 | Belady_MIN | 0.2 | 0.4 | 1.000 | n/a |
| B_temporal_locality | 200 | FIFO | 6.2 | 3.9 | 0.997 | 6.00 |
| B_temporal_locality | 200 | LRU | 1.6 | 1.9 | 0.999 | 1.47 |
| B_temporal_locality | 200 | LFU | 18.4 | 14.6 | 0.991 | 18.20 |
| B_temporal_locality | 200 | Random | 8.7 | 3.6 | 0.996 | 8.51 |
| B_temporal_locality | 200 | Learned | 1.7 | 2.0 | 0.999 | 1.57 |
| B_temporal_locality | 500 | Belady_MIN | 0.2 | 0.4 | 1.000 | n/a |
| B_temporal_locality | 500 | FIFO | 6.2 | 3.9 | 0.997 | 6.00 |
| B_temporal_locality | 500 | LRU | 1.6 | 1.9 | 0.999 | 1.47 |
| B_temporal_locality | 500 | LFU | 18.4 | 14.6 | 0.991 | 18.20 |
| B_temporal_locality | 500 | Random | 8.7 | 3.6 | 0.996 | 8.51 |
| B_temporal_locality | 500 | Learned | 1.7 | 1.9 | 0.999 | 1.53 |
| C_bursty | 25 | Belady_MIN | 13.7 | 4.3 | 0.993 | n/a |
| C_bursty | 25 | FIFO | 45.7 | 8.4 | 0.977 | 2.53 |
| C_bursty | 25 | LRU | 45.3 | 8.5 | 0.977 | 2.50 |
| C_bursty | 25 | LFU | 45.6 | 9.0 | 0.977 | 2.56 |
| C_bursty | 25 | Random | 45.9 | 7.9 | 0.977 | 2.55 |
| C_bursty | 25 | Learned | 47.1 | 8.3 | 0.976 | 2.70 |
| C_bursty | 50 | Belady_MIN | 13.7 | 4.3 | 0.993 | n/a |
| C_bursty | 50 | FIFO | 45.7 | 8.4 | 0.977 | 2.53 |
| C_bursty | 50 | LRU | 45.3 | 8.5 | 0.977 | 2.50 |
| C_bursty | 50 | LFU | 45.6 | 9.0 | 0.977 | 2.56 |
| C_bursty | 50 | Random | 45.9 | 7.9 | 0.977 | 2.55 |
| C_bursty | 50 | Learned | 45.8 | 7.5 | 0.977 | 2.57 |
| C_bursty | 100 | Belady_MIN | 13.7 | 4.3 | 0.993 | n/a |
| C_bursty | 100 | FIFO | 45.7 | 8.4 | 0.977 | 2.53 |
| C_bursty | 100 | LRU | 45.3 | 8.5 | 0.977 | 2.50 |
| C_bursty | 100 | LFU | 45.6 | 9.0 | 0.977 | 2.56 |
| C_bursty | 100 | Random | 45.9 | 7.9 | 0.977 | 2.55 |
| C_bursty | 100 | Learned | 45.6 | 8.9 | 0.977 | 2.53 |
| C_bursty | 200 | Belady_MIN | 13.7 | 4.3 | 0.993 | n/a |
| C_bursty | 200 | FIFO | 45.7 | 8.4 | 0.977 | 2.53 |
| C_bursty | 200 | LRU | 45.3 | 8.5 | 0.977 | 2.50 |
| C_bursty | 200 | LFU | 45.6 | 9.0 | 0.977 | 2.56 |
| C_bursty | 200 | Random | 45.9 | 7.9 | 0.977 | 2.55 |
| C_bursty | 200 | Learned | 46.5 | 8.2 | 0.977 | 2.61 |
| C_bursty | 500 | Belady_MIN | 13.7 | 4.3 | 0.993 | n/a |
| C_bursty | 500 | FIFO | 45.7 | 8.4 | 0.977 | 2.53 |
| C_bursty | 500 | LRU | 45.3 | 8.5 | 0.977 | 2.50 |
| C_bursty | 500 | LFU | 45.6 | 9.0 | 0.977 | 2.56 |
| C_bursty | 500 | Random | 45.9 | 7.9 | 0.977 | 2.55 |
| C_bursty | 500 | Learned | 45.6 | 9.2 | 0.977 | 2.51 |
| D_long_range_reuse [structural-null] | 25 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [structural-null] | 25 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 25 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 25 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 25 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 25 | Learned | n/a | n/a | n/a | n/a (model not trained -- one-class training data) |
| D_long_range_reuse [structural-null] | 50 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [structural-null] | 50 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 50 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 50 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 50 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 50 | Learned | n/a | n/a | n/a | n/a (model not trained -- one-class training data) |
| D_long_range_reuse [structural-null] | 100 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [structural-null] | 100 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 100 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 100 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 100 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 100 | Learned | n/a | n/a | n/a | n/a (model not trained -- one-class training data) |
| D_long_range_reuse [redundant w/ H=500] | 200 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [redundant w/ H=500] | 200 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=500] | 200 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=500] | 200 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=500] | 200 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=500] | 200 | Learned | 1770.0 | 0.0 | 0.115 | 0.07 |
| D_long_range_reuse [redundant w/ H=200] | 500 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [redundant w/ H=200] | 500 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=200] | 500 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=200] | 500 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=200] | 500 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=200] | 500 | Learned | 1770.0 | 0.0 | 0.115 | 0.07 |
| E_phase_changing | 25 | Belady_MIN | 306.8 | 11.6 | 0.847 | n/a |
| E_phase_changing | 25 | FIFO | 782.4 | 24.6 | 0.609 | 1.55 |
| E_phase_changing | 25 | LRU | 787.5 | 26.0 | 0.606 | 1.57 |
| E_phase_changing | 25 | LFU | 1518.6 | 10.3 | 0.241 | 3.96 |
| E_phase_changing | 25 | Random | 822.6 | 16.6 | 0.589 | 1.68 |
| E_phase_changing | 25 | Learned | 781.8 | 25.9 | 0.609 | 1.55 |
| E_phase_changing | 50 | Belady_MIN | 306.8 | 11.6 | 0.847 | n/a |
| E_phase_changing | 50 | FIFO | 782.4 | 24.6 | 0.609 | 1.55 |
| E_phase_changing | 50 | LRU | 787.5 | 26.0 | 0.606 | 1.57 |
| E_phase_changing | 50 | LFU | 1518.6 | 10.3 | 0.241 | 3.96 |
| E_phase_changing | 50 | Random | 822.6 | 16.6 | 0.589 | 1.68 |
| E_phase_changing | 50 | Learned | 779.7 | 25.8 | 0.610 | 1.54 |
| E_phase_changing | 100 | Belady_MIN | 306.8 | 11.6 | 0.847 | n/a |
| E_phase_changing | 100 | FIFO | 782.4 | 24.6 | 0.609 | 1.55 |
| E_phase_changing | 100 | LRU | 787.5 | 26.0 | 0.606 | 1.57 |
| E_phase_changing | 100 | LFU | 1518.6 | 10.3 | 0.241 | 3.96 |
| E_phase_changing | 100 | Random | 822.6 | 16.6 | 0.589 | 1.68 |
| E_phase_changing | 100 | Learned | 779.4 | 27.9 | 0.610 | 1.54 |
| E_phase_changing [redundant w/ H=500] | 200 | Belady_MIN | 306.8 | 11.6 | 0.847 | n/a |
| E_phase_changing [redundant w/ H=500] | 200 | FIFO | 782.4 | 24.6 | 0.609 | 1.55 |
| E_phase_changing [redundant w/ H=500] | 200 | LRU | 787.5 | 26.0 | 0.606 | 1.57 |
| E_phase_changing [redundant w/ H=500] | 200 | LFU | 1518.6 | 10.3 | 0.241 | 3.96 |
| E_phase_changing [redundant w/ H=500] | 200 | Random | 822.6 | 16.6 | 0.589 | 1.68 |
| E_phase_changing [redundant w/ H=500] | 200 | Learned | 784.6 | 27.1 | 0.608 | 1.56 |
| E_phase_changing [redundant w/ H=200] | 500 | Belady_MIN | 306.8 | 11.6 | 0.847 | n/a |
| E_phase_changing [redundant w/ H=200] | 500 | FIFO | 782.4 | 24.6 | 0.609 | 1.55 |
| E_phase_changing [redundant w/ H=200] | 500 | LRU | 787.5 | 26.0 | 0.606 | 1.57 |
| E_phase_changing [redundant w/ H=200] | 500 | LFU | 1518.6 | 10.3 | 0.241 | 3.96 |
| E_phase_changing [redundant w/ H=200] | 500 | Random | 822.6 | 16.6 | 0.589 | 1.68 |
| E_phase_changing [redundant w/ H=200] | 500 | Learned | 784.1 | 26.1 | 0.608 | 1.56 |
| F_non_stationary | 25 | Belady_MIN | 401.7 | 11.3 | 0.799 | n/a |
| F_non_stationary | 25 | FIFO | 881.5 | 25.3 | 0.559 | 1.19 |
| F_non_stationary | 25 | LRU | 770.4 | 22.0 | 0.615 | 0.92 |
| F_non_stationary | 25 | LFU | 908.5 | 17.6 | 0.546 | 1.26 |
| F_non_stationary | 25 | Random | 885.8 | 21.7 | 0.557 | 1.21 |
| F_non_stationary | 25 | Learned | 617.1 | 18.7 | 0.691 | 0.54 |
| F_non_stationary | 50 | Belady_MIN | 401.7 | 11.3 | 0.799 | n/a |
| F_non_stationary | 50 | FIFO | 881.5 | 25.3 | 0.559 | 1.19 |
| F_non_stationary | 50 | LRU | 770.4 | 22.0 | 0.615 | 0.92 |
| F_non_stationary | 50 | LFU | 908.5 | 17.6 | 0.546 | 1.26 |
| F_non_stationary | 50 | Random | 885.8 | 21.7 | 0.557 | 1.21 |
| F_non_stationary | 50 | Learned | 620.1 | 17.4 | 0.690 | 0.54 |
| F_non_stationary | 100 | Belady_MIN | 401.7 | 11.3 | 0.799 | n/a |
| F_non_stationary | 100 | FIFO | 881.5 | 25.3 | 0.559 | 1.19 |
| F_non_stationary | 100 | LRU | 770.4 | 22.0 | 0.615 | 0.92 |
| F_non_stationary | 100 | LFU | 908.5 | 17.6 | 0.546 | 1.26 |
| F_non_stationary | 100 | Random | 885.8 | 21.7 | 0.557 | 1.21 |
| F_non_stationary | 100 | Learned | 631.4 | 21.0 | 0.684 | 0.57 |
| F_non_stationary | 200 | Belady_MIN | 401.7 | 11.3 | 0.799 | n/a |
| F_non_stationary | 200 | FIFO | 881.5 | 25.3 | 0.559 | 1.19 |
| F_non_stationary | 200 | LRU | 770.4 | 22.0 | 0.615 | 0.92 |
| F_non_stationary | 200 | LFU | 908.5 | 17.6 | 0.546 | 1.26 |
| F_non_stationary | 200 | Random | 885.8 | 21.7 | 0.557 | 1.21 |
| F_non_stationary | 200 | Learned | 656.9 | 20.5 | 0.672 | 0.64 |
| F_non_stationary | 500 | Belady_MIN | 401.7 | 11.3 | 0.799 | n/a |
| F_non_stationary | 500 | FIFO | 881.5 | 25.3 | 0.559 | 1.19 |
| F_non_stationary | 500 | LRU | 770.4 | 22.0 | 0.615 | 0.92 |
| F_non_stationary | 500 | LFU | 908.5 | 17.6 | 0.546 | 1.26 |
| F_non_stationary | 500 | Random | 885.8 | 21.7 | 0.557 | 1.21 |
| F_non_stationary | 500 | Learned | 749.3 | 29.1 | 0.625 | 0.87 |
