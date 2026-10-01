Table 10 -- Protocol B (leave-one-family-out) operational results, same structure as Table 9, one block per held-out family. All test episodes are 100% of the held-out family's episodes, never seen in training or validation. Source: protocol_b_*.json checkpoints.

| Held-out family | H | Policy | Mean faults | Std | Hit ratio | Belady gap |
| --- | --- | --- | --- | --- | --- | --- |
| A_uniform_random | 25 | Belady_MIN | 813.1 | 11.9 | 0.593 | n/a |
| A_uniform_random | 25 | FIFO | 1437.7 | 19.4 | 0.281 | 0.77 |
| A_uniform_random | 25 | LRU | 1437.6 | 18.7 | 0.281 | 0.77 |
| A_uniform_random | 25 | LFU | 1438.7 | 17.5 | 0.281 | 0.77 |
| A_uniform_random | 25 | Random | 1437.5 | 14.4 | 0.281 | 0.77 |
| A_uniform_random | 25 | Learned | 1437.9 | 17.8 | 0.281 | 0.77 |
| A_uniform_random | 50 | Belady_MIN | 813.1 | 11.9 | 0.593 | n/a |
| A_uniform_random | 50 | FIFO | 1437.7 | 19.4 | 0.281 | 0.77 |
| A_uniform_random | 50 | LRU | 1437.6 | 18.7 | 0.281 | 0.77 |
| A_uniform_random | 50 | LFU | 1438.7 | 17.5 | 0.281 | 0.77 |
| A_uniform_random | 50 | Random | 1437.5 | 14.4 | 0.281 | 0.77 |
| A_uniform_random | 50 | Learned | 1438.7 | 18.2 | 0.281 | 0.77 |
| A_uniform_random | 100 | Belady_MIN | 813.1 | 11.9 | 0.593 | n/a |
| A_uniform_random | 100 | FIFO | 1437.7 | 19.4 | 0.281 | 0.77 |
| A_uniform_random | 100 | LRU | 1437.6 | 18.7 | 0.281 | 0.77 |
| A_uniform_random | 100 | LFU | 1438.7 | 17.5 | 0.281 | 0.77 |
| A_uniform_random | 100 | Random | 1437.5 | 14.4 | 0.281 | 0.77 |
| A_uniform_random | 100 | Learned | 1438.6 | 19.0 | 0.281 | 0.77 |
| A_uniform_random | 200 | Belady_MIN | 813.1 | 11.9 | 0.593 | n/a |
| A_uniform_random | 200 | FIFO | 1437.7 | 19.4 | 0.281 | 0.77 |
| A_uniform_random | 200 | LRU | 1437.6 | 18.7 | 0.281 | 0.77 |
| A_uniform_random | 200 | LFU | 1438.7 | 17.5 | 0.281 | 0.77 |
| A_uniform_random | 200 | Random | 1437.5 | 14.4 | 0.281 | 0.77 |
| A_uniform_random | 200 | Learned | 1438.0 | 17.7 | 0.281 | 0.77 |
| A_uniform_random | 500 | Belady_MIN | 813.1 | 11.9 | 0.593 | n/a |
| A_uniform_random | 500 | FIFO | 1437.7 | 19.4 | 0.281 | 0.77 |
| A_uniform_random | 500 | LRU | 1437.6 | 18.7 | 0.281 | 0.77 |
| A_uniform_random | 500 | LFU | 1438.7 | 17.5 | 0.281 | 0.77 |
| A_uniform_random | 500 | Random | 1437.5 | 14.4 | 0.281 | 0.77 |
| A_uniform_random | 500 | Learned | 1437.4 | 18.5 | 0.281 | 0.77 |
| B_temporal_locality | 25 | Belady_MIN | 0.2 | 0.5 | 1.000 | n/a |
| B_temporal_locality | 25 | FIFO | 6.4 | 4.0 | 0.997 | 6.08 |
| B_temporal_locality | 25 | LRU | 1.7 | 1.7 | 0.999 | 1.40 |
| B_temporal_locality | 25 | LFU | 17.6 | 16.9 | 0.991 | 16.97 |
| B_temporal_locality | 25 | Random | 8.8 | 4.5 | 0.996 | 8.37 |
| B_temporal_locality | 25 | Learned | 7.5 | 5.5 | 0.996 | 7.21 |
| B_temporal_locality | 50 | Belady_MIN | 0.2 | 0.5 | 1.000 | n/a |
| B_temporal_locality | 50 | FIFO | 6.4 | 4.0 | 0.997 | 6.08 |
| B_temporal_locality | 50 | LRU | 1.7 | 1.7 | 0.999 | 1.40 |
| B_temporal_locality | 50 | LFU | 17.6 | 16.9 | 0.991 | 16.97 |
| B_temporal_locality | 50 | Random | 8.8 | 4.5 | 0.996 | 8.37 |
| B_temporal_locality | 50 | Learned | 8.3 | 6.0 | 0.996 | 7.88 |
| B_temporal_locality | 100 | Belady_MIN | 0.2 | 0.5 | 1.000 | n/a |
| B_temporal_locality | 100 | FIFO | 6.4 | 4.0 | 0.997 | 6.08 |
| B_temporal_locality | 100 | LRU | 1.7 | 1.7 | 0.999 | 1.40 |
| B_temporal_locality | 100 | LFU | 17.6 | 16.9 | 0.991 | 16.97 |
| B_temporal_locality | 100 | Random | 8.8 | 4.5 | 0.996 | 8.37 |
| B_temporal_locality | 100 | Learned | 7.2 | 6.1 | 0.996 | 6.86 |
| B_temporal_locality | 200 | Belady_MIN | 0.2 | 0.5 | 1.000 | n/a |
| B_temporal_locality | 200 | FIFO | 6.4 | 4.0 | 0.997 | 6.08 |
| B_temporal_locality | 200 | LRU | 1.7 | 1.7 | 0.999 | 1.40 |
| B_temporal_locality | 200 | LFU | 17.6 | 16.9 | 0.991 | 16.97 |
| B_temporal_locality | 200 | Random | 8.8 | 4.5 | 0.996 | 8.37 |
| B_temporal_locality | 200 | Learned | 6.5 | 4.9 | 0.997 | 6.20 |
| B_temporal_locality | 500 | Belady_MIN | 0.2 | 0.5 | 1.000 | n/a |
| B_temporal_locality | 500 | FIFO | 6.4 | 4.0 | 0.997 | 6.08 |
| B_temporal_locality | 500 | LRU | 1.7 | 1.7 | 0.999 | 1.40 |
| B_temporal_locality | 500 | LFU | 17.6 | 16.9 | 0.991 | 16.97 |
| B_temporal_locality | 500 | Random | 8.8 | 4.5 | 0.996 | 8.37 |
| B_temporal_locality | 500 | Learned | 7.7 | 5.5 | 0.996 | 7.29 |
| C_bursty | 25 | Belady_MIN | 14.8 | 4.5 | 0.993 | n/a |
| C_bursty | 25 | FIFO | 47.5 | 8.6 | 0.976 | 2.41 |
| C_bursty | 25 | LRU | 47.5 | 8.2 | 0.976 | 2.42 |
| C_bursty | 25 | LFU | 47.5 | 9.1 | 0.976 | 2.42 |
| C_bursty | 25 | Random | 47.2 | 8.0 | 0.976 | 2.39 |
| C_bursty | 25 | Learned | 47.0 | 8.8 | 0.977 | 2.35 |
| C_bursty | 50 | Belady_MIN | 14.8 | 4.5 | 0.993 | n/a |
| C_bursty | 50 | FIFO | 47.5 | 8.6 | 0.976 | 2.41 |
| C_bursty | 50 | LRU | 47.5 | 8.2 | 0.976 | 2.42 |
| C_bursty | 50 | LFU | 47.5 | 9.1 | 0.976 | 2.42 |
| C_bursty | 50 | Random | 47.2 | 8.0 | 0.976 | 2.39 |
| C_bursty | 50 | Learned | 47.2 | 8.5 | 0.976 | 2.38 |
| C_bursty | 100 | Belady_MIN | 14.8 | 4.5 | 0.993 | n/a |
| C_bursty | 100 | FIFO | 47.5 | 8.6 | 0.976 | 2.41 |
| C_bursty | 100 | LRU | 47.5 | 8.2 | 0.976 | 2.42 |
| C_bursty | 100 | LFU | 47.5 | 9.1 | 0.976 | 2.42 |
| C_bursty | 100 | Random | 47.2 | 8.0 | 0.976 | 2.39 |
| C_bursty | 100 | Learned | 47.3 | 8.7 | 0.976 | 2.40 |
| C_bursty | 200 | Belady_MIN | 14.8 | 4.5 | 0.993 | n/a |
| C_bursty | 200 | FIFO | 47.5 | 8.6 | 0.976 | 2.41 |
| C_bursty | 200 | LRU | 47.5 | 8.2 | 0.976 | 2.42 |
| C_bursty | 200 | LFU | 47.5 | 9.1 | 0.976 | 2.42 |
| C_bursty | 200 | Random | 47.2 | 8.0 | 0.976 | 2.39 |
| C_bursty | 200 | Learned | 47.2 | 8.6 | 0.976 | 2.39 |
| C_bursty | 500 | Belady_MIN | 14.8 | 4.5 | 0.993 | n/a |
| C_bursty | 500 | FIFO | 47.5 | 8.6 | 0.976 | 2.41 |
| C_bursty | 500 | LRU | 47.5 | 8.2 | 0.976 | 2.42 |
| C_bursty | 500 | LFU | 47.5 | 9.1 | 0.976 | 2.42 |
| C_bursty | 500 | Random | 47.2 | 8.0 | 0.976 | 2.39 |
| C_bursty | 500 | Learned | 47.3 | 8.4 | 0.976 | 2.40 |
| D_long_range_reuse [structural-null] | 25 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [structural-null] | 25 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 25 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 25 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 25 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 25 | Learned | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 50 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [structural-null] | 50 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 50 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 50 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 50 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 50 | Learned | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 100 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [structural-null] | 100 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 100 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 100 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 100 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [structural-null] | 100 | Learned | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=500] | 200 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [redundant w/ H=500] | 200 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=500] | 200 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=500] | 200 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=500] | 200 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=500] | 200 | Learned | 1795.0 | 0.0 | 0.102 | 0.08 |
| D_long_range_reuse [redundant w/ H=200] | 500 | Belady_MIN | 1656.0 | 0.0 | 0.172 | n/a |
| D_long_range_reuse [redundant w/ H=200] | 500 | FIFO | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=200] | 500 | LRU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=200] | 500 | LFU | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=200] | 500 | Random | 1800.0 | 0.0 | 0.100 | 0.09 |
| D_long_range_reuse [redundant w/ H=200] | 500 | Learned | 1797.0 | 0.0 | 0.102 | 0.09 |
| E_phase_changing | 25 | Belady_MIN | 306.6 | 10.3 | 0.847 | n/a |
| E_phase_changing | 25 | FIFO | 782.6 | 22.2 | 0.609 | 1.55 |
| E_phase_changing | 25 | LRU | 784.7 | 23.1 | 0.608 | 1.56 |
| E_phase_changing | 25 | LFU | 1520.4 | 11.4 | 0.240 | 3.96 |
| E_phase_changing | 25 | Random | 823.5 | 14.9 | 0.588 | 1.69 |
| E_phase_changing | 25 | Learned | 856.9 | 22.5 | 0.572 | 1.80 |
| E_phase_changing | 50 | Belady_MIN | 306.6 | 10.3 | 0.847 | n/a |
| E_phase_changing | 50 | FIFO | 782.6 | 22.2 | 0.609 | 1.55 |
| E_phase_changing | 50 | LRU | 784.7 | 23.1 | 0.608 | 1.56 |
| E_phase_changing | 50 | LFU | 1520.4 | 11.4 | 0.240 | 3.96 |
| E_phase_changing | 50 | Random | 823.5 | 14.9 | 0.588 | 1.69 |
| E_phase_changing | 50 | Learned | 859.9 | 22.7 | 0.570 | 1.81 |
| E_phase_changing | 100 | Belady_MIN | 306.6 | 10.3 | 0.847 | n/a |
| E_phase_changing | 100 | FIFO | 782.6 | 22.2 | 0.609 | 1.55 |
| E_phase_changing | 100 | LRU | 784.7 | 23.1 | 0.608 | 1.56 |
| E_phase_changing | 100 | LFU | 1520.4 | 11.4 | 0.240 | 3.96 |
| E_phase_changing | 100 | Random | 823.5 | 14.9 | 0.588 | 1.69 |
| E_phase_changing | 100 | Learned | 861.4 | 21.9 | 0.569 | 1.81 |
| E_phase_changing [redundant w/ H=500] | 200 | Belady_MIN | 306.6 | 10.3 | 0.847 | n/a |
| E_phase_changing [redundant w/ H=500] | 200 | FIFO | 782.6 | 22.2 | 0.609 | 1.55 |
| E_phase_changing [redundant w/ H=500] | 200 | LRU | 784.7 | 23.1 | 0.608 | 1.56 |
| E_phase_changing [redundant w/ H=500] | 200 | LFU | 1520.4 | 11.4 | 0.240 | 3.96 |
| E_phase_changing [redundant w/ H=500] | 200 | Random | 823.5 | 14.9 | 0.588 | 1.69 |
| E_phase_changing [redundant w/ H=500] | 200 | Learned | 808.2 | 23.3 | 0.596 | 1.64 |
| E_phase_changing [redundant w/ H=200] | 500 | Belady_MIN | 306.6 | 10.3 | 0.847 | n/a |
| E_phase_changing [redundant w/ H=200] | 500 | FIFO | 782.6 | 22.2 | 0.609 | 1.55 |
| E_phase_changing [redundant w/ H=200] | 500 | LRU | 784.7 | 23.1 | 0.608 | 1.56 |
| E_phase_changing [redundant w/ H=200] | 500 | LFU | 1520.4 | 11.4 | 0.240 | 3.96 |
| E_phase_changing [redundant w/ H=200] | 500 | Random | 823.5 | 14.9 | 0.588 | 1.69 |
| E_phase_changing [redundant w/ H=200] | 500 | Learned | 859.2 | 24.2 | 0.570 | 1.80 |
| F_non_stationary | 25 | Belady_MIN | 401.8 | 14.4 | 0.799 | n/a |
| F_non_stationary | 25 | FIFO | 880.6 | 25.4 | 0.560 | 1.19 |
| F_non_stationary | 25 | LRU | 769.5 | 26.2 | 0.615 | 0.92 |
| F_non_stationary | 25 | LFU | 906.8 | 22.5 | 0.547 | 1.26 |
| F_non_stationary | 25 | Random | 885.0 | 23.1 | 0.558 | 1.20 |
| F_non_stationary | 25 | Learned | 870.7 | 29.2 | 0.565 | 1.17 |
| F_non_stationary | 50 | Belady_MIN | 401.8 | 14.4 | 0.799 | n/a |
| F_non_stationary | 50 | FIFO | 880.6 | 25.4 | 0.560 | 1.19 |
| F_non_stationary | 50 | LRU | 769.5 | 26.2 | 0.615 | 0.92 |
| F_non_stationary | 50 | LFU | 906.8 | 22.5 | 0.547 | 1.26 |
| F_non_stationary | 50 | Random | 885.0 | 23.1 | 0.558 | 1.20 |
| F_non_stationary | 50 | Learned | 888.0 | 28.6 | 0.556 | 1.21 |
| F_non_stationary | 100 | Belady_MIN | 401.8 | 14.4 | 0.799 | n/a |
| F_non_stationary | 100 | FIFO | 880.6 | 25.4 | 0.560 | 1.19 |
| F_non_stationary | 100 | LRU | 769.5 | 26.2 | 0.615 | 0.92 |
| F_non_stationary | 100 | LFU | 906.8 | 22.5 | 0.547 | 1.26 |
| F_non_stationary | 100 | Random | 885.0 | 23.1 | 0.558 | 1.20 |
| F_non_stationary | 100 | Learned | 849.2 | 31.5 | 0.575 | 1.11 |
| F_non_stationary | 200 | Belady_MIN | 401.8 | 14.4 | 0.799 | n/a |
| F_non_stationary | 200 | FIFO | 880.6 | 25.4 | 0.560 | 1.19 |
| F_non_stationary | 200 | LRU | 769.5 | 26.2 | 0.615 | 0.92 |
| F_non_stationary | 200 | LFU | 906.8 | 22.5 | 0.547 | 1.26 |
| F_non_stationary | 200 | Random | 885.0 | 23.1 | 0.558 | 1.20 |
| F_non_stationary | 200 | Learned | 851.9 | 30.6 | 0.574 | 1.12 |
| F_non_stationary | 500 | Belady_MIN | 401.8 | 14.4 | 0.799 | n/a |
| F_non_stationary | 500 | FIFO | 880.6 | 25.4 | 0.560 | 1.19 |
| F_non_stationary | 500 | LRU | 769.5 | 26.2 | 0.615 | 0.92 |
| F_non_stationary | 500 | LFU | 906.8 | 22.5 | 0.547 | 1.26 |
| F_non_stationary | 500 | Random | 885.0 | 23.1 | 0.558 | 1.20 |
| F_non_stationary | 500 | Learned | 1025.3 | 40.5 | 0.487 | 1.55 |
