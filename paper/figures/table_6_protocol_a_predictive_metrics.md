Table 6 -- Protocol A (within-distribution) predictive metrics, per family, horizon, model, and split. Source: experiments/generalization-experiment-2/checkpoints/protocol_a_*.json (the `models.<name>.{train,val,test}_metrics` fields, read verbatim). n/a = ROC-AUC/PR-AUC undefined on a structurally one-class split (see flags).

| Family | H | Model | Split | ROC-AUC | PR-AUC | Brier | F1 | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_uniform_random | 25 | logistic_regression | train | 0.505 | 0.327 | 0.219 | 0.000 | 3324720 |
| A_uniform_random | 25 | logistic_regression | val | 0.504 | 0.326 | 0.219 | 0.000 | 715312 |
| A_uniform_random | 25 | logistic_regression | test | 0.500 | 0.323 | 0.219 | 0.000 | 713888 |
| A_uniform_random | 25 | hist_gradient_boosting | train | 0.571 | 0.411 | 0.217 | 0.003 | 3324720 |
| A_uniform_random | 25 | hist_gradient_boosting | val | 0.500 | 0.322 | 0.219 | 0.001 | 715312 |
| A_uniform_random | 25 | hist_gradient_boosting | test | 0.502 | 0.324 | 0.219 | 0.001 | 713888 |
| A_uniform_random | 50 | logistic_regression | train | 0.510 | 0.544 | 0.248 | 0.700 | 3324720 |
| A_uniform_random | 50 | logistic_regression | val | 0.511 | 0.543 | 0.249 | 0.699 | 715312 |
| A_uniform_random | 50 | logistic_regression | test | 0.511 | 0.542 | 0.248 | 0.699 | 713888 |
| A_uniform_random | 50 | hist_gradient_boosting | train | 0.574 | 0.614 | 0.245 | 0.704 | 3324720 |
| A_uniform_random | 50 | hist_gradient_boosting | val | 0.507 | 0.540 | 0.248 | 0.694 | 715312 |
| A_uniform_random | 50 | hist_gradient_boosting | test | 0.513 | 0.544 | 0.248 | 0.694 | 713888 |
| A_uniform_random | 100 | logistic_regression | train | 0.538 | 0.791 | 0.172 | 0.875 | 3324720 |
| A_uniform_random | 100 | logistic_regression | val | 0.543 | 0.794 | 0.172 | 0.875 | 715312 |
| A_uniform_random | 100 | logistic_regression | test | 0.537 | 0.791 | 0.172 | 0.876 | 713888 |
| A_uniform_random | 100 | hist_gradient_boosting | train | 0.598 | 0.829 | 0.166 | 0.877 | 3324720 |
| A_uniform_random | 100 | hist_gradient_boosting | val | 0.543 | 0.794 | 0.168 | 0.877 | 715312 |
| A_uniform_random | 100 | hist_gradient_boosting | test | 0.540 | 0.792 | 0.168 | 0.877 | 713888 |
| A_uniform_random | 200 | logistic_regression | train | 0.689 | 0.957 | 0.061 | 0.964 | 3324720 |
| A_uniform_random | 200 | logistic_regression | val | 0.689 | 0.956 | 0.061 | 0.964 | 715312 |
| A_uniform_random | 200 | logistic_regression | test | 0.686 | 0.956 | 0.061 | 0.964 | 713888 |
| A_uniform_random | 200 | hist_gradient_boosting | train | 0.745 | 0.968 | 0.053 | 0.968 | 3324720 |
| A_uniform_random | 200 | hist_gradient_boosting | val | 0.692 | 0.957 | 0.054 | 0.967 | 715312 |
| A_uniform_random | 200 | hist_gradient_boosting | test | 0.689 | 0.956 | 0.055 | 0.967 | 713888 |
| A_uniform_random | 500 | logistic_regression | train | 0.970 | 0.999 | 0.020 | 0.986 | 3324720 |
| A_uniform_random | 500 | logistic_regression | val | 0.974 | 0.999 | 0.019 | 0.986 | 715312 |
| A_uniform_random | 500 | logistic_regression | test | 0.969 | 0.999 | 0.020 | 0.986 | 713888 |
| A_uniform_random | 500 | hist_gradient_boosting | train | 0.977 | 0.999 | 0.018 | 0.987 | 3324720 |
| A_uniform_random | 500 | hist_gradient_boosting | val | 0.972 | 0.999 | 0.020 | 0.986 | 715312 |
| A_uniform_random | 500 | hist_gradient_boosting | test | 0.968 | 0.999 | 0.020 | 0.986 | 713888 |
| B_temporal_locality | 25 | logistic_regression | train | 0.845 | 0.722 | 0.153 | 0.662 | 9664 |
| B_temporal_locality | 25 | logistic_regression | val | 0.846 | 0.727 | 0.154 | 0.656 | 1888 |
| B_temporal_locality | 25 | logistic_regression | test | 0.844 | 0.717 | 0.154 | 0.670 | 2016 |
| B_temporal_locality | 25 | hist_gradient_boosting | train | 0.930 | 0.885 | 0.107 | 0.779 | 9664 |
| B_temporal_locality | 25 | hist_gradient_boosting | val | 0.846 | 0.730 | 0.150 | 0.694 | 1888 |
| B_temporal_locality | 25 | hist_gradient_boosting | test | 0.850 | 0.743 | 0.147 | 0.698 | 2016 |
| B_temporal_locality | 50 | logistic_regression | train | 0.847 | 0.803 | 0.160 | 0.745 | 9664 |
| B_temporal_locality | 50 | logistic_regression | val | 0.835 | 0.797 | 0.166 | 0.736 | 1888 |
| B_temporal_locality | 50 | logistic_regression | test | 0.835 | 0.793 | 0.165 | 0.740 | 2016 |
| B_temporal_locality | 50 | hist_gradient_boosting | train | 0.927 | 0.919 | 0.113 | 0.824 | 9664 |
| B_temporal_locality | 50 | hist_gradient_boosting | val | 0.835 | 0.798 | 0.164 | 0.751 | 1888 |
| B_temporal_locality | 50 | hist_gradient_boosting | test | 0.836 | 0.803 | 0.163 | 0.746 | 2016 |
| B_temporal_locality | 100 | logistic_regression | train | 0.849 | 0.871 | 0.158 | 0.805 | 9664 |
| B_temporal_locality | 100 | logistic_regression | val | 0.834 | 0.866 | 0.165 | 0.801 | 1888 |
| B_temporal_locality | 100 | logistic_regression | test | 0.832 | 0.851 | 0.166 | 0.791 | 2016 |
| B_temporal_locality | 100 | hist_gradient_boosting | train | 0.930 | 0.949 | 0.110 | 0.863 | 9664 |
| B_temporal_locality | 100 | hist_gradient_boosting | val | 0.839 | 0.869 | 0.159 | 0.806 | 1888 |
| B_temporal_locality | 100 | hist_gradient_boosting | test | 0.830 | 0.851 | 0.168 | 0.792 | 2016 |
| B_temporal_locality | 200 | logistic_regression | train | 0.839 | 0.903 | 0.154 | 0.844 | 9664 |
| B_temporal_locality | 200 | logistic_regression | val | 0.821 | 0.893 | 0.160 | 0.833 | 1888 |
| B_temporal_locality | 200 | logistic_regression | test | 0.823 | 0.894 | 0.160 | 0.829 | 2016 |
| B_temporal_locality | 200 | hist_gradient_boosting | train | 0.933 | 0.966 | 0.103 | 0.893 | 9664 |
| B_temporal_locality | 200 | hist_gradient_boosting | val | 0.818 | 0.893 | 0.161 | 0.834 | 1888 |
| B_temporal_locality | 200 | hist_gradient_boosting | test | 0.817 | 0.894 | 0.163 | 0.824 | 2016 |
| B_temporal_locality | 500 | logistic_regression | train | 0.830 | 0.933 | 0.137 | 0.876 | 9664 |
| B_temporal_locality | 500 | logistic_regression | val | 0.807 | 0.924 | 0.143 | 0.872 | 1888 |
| B_temporal_locality | 500 | logistic_regression | test | 0.808 | 0.928 | 0.141 | 0.873 | 2016 |
| B_temporal_locality | 500 | hist_gradient_boosting | train | 0.953 | 0.984 | 0.081 | 0.934 | 9664 |
| B_temporal_locality | 500 | hist_gradient_boosting | val | 0.799 | 0.920 | 0.146 | 0.865 | 1888 |
| B_temporal_locality | 500 | hist_gradient_boosting | test | 0.797 | 0.925 | 0.148 | 0.858 | 2016 |
| C_bursty | 25 | logistic_regression | train | 0.518 | 0.027 | 0.024 | 0.000 | 196944 |
| C_bursty | 25 | logistic_regression | val | 0.490 | 0.025 | 0.025 | 0.000 | 42416 |
| C_bursty | 25 | logistic_regression | test | 0.478 | 0.024 | 0.025 | 0.000 | 40928 |
| C_bursty | 25 | hist_gradient_boosting | train | 0.773 | 0.271 | 0.023 | 0.009 | 196944 |
| C_bursty | 25 | hist_gradient_boosting | val | 0.467 | 0.023 | 0.025 | 0.000 | 42416 |
| C_bursty | 25 | hist_gradient_boosting | test | 0.502 | 0.025 | 0.025 | 0.000 | 40928 |
| C_bursty | 50 | logistic_regression | train | 0.515 | 0.052 | 0.047 | 0.000 | 196944 |
| C_bursty | 50 | logistic_regression | val | 0.480 | 0.049 | 0.049 | 0.000 | 42416 |
| C_bursty | 50 | logistic_regression | test | 0.491 | 0.049 | 0.049 | 0.000 | 40928 |
| C_bursty | 50 | hist_gradient_boosting | train | 0.745 | 0.295 | 0.043 | 0.020 | 196944 |
| C_bursty | 50 | hist_gradient_boosting | val | 0.491 | 0.050 | 0.050 | 0.000 | 42416 |
| C_bursty | 50 | hist_gradient_boosting | test | 0.518 | 0.053 | 0.049 | 0.000 | 40928 |
| C_bursty | 100 | logistic_regression | train | 0.510 | 0.097 | 0.085 | 0.000 | 196944 |
| C_bursty | 100 | logistic_regression | val | 0.499 | 0.098 | 0.089 | 0.000 | 42416 |
| C_bursty | 100 | logistic_regression | test | 0.497 | 0.096 | 0.088 | 0.000 | 40928 |
| C_bursty | 100 | hist_gradient_boosting | train | 0.722 | 0.337 | 0.080 | 0.029 | 196944 |
| C_bursty | 100 | hist_gradient_boosting | val | 0.490 | 0.094 | 0.090 | 0.000 | 42416 |
| C_bursty | 100 | hist_gradient_boosting | test | 0.518 | 0.102 | 0.089 | 0.001 | 40928 |
| C_bursty | 200 | logistic_regression | train | 0.513 | 0.180 | 0.145 | 0.000 | 196944 |
| C_bursty | 200 | logistic_regression | val | 0.512 | 0.184 | 0.146 | 0.000 | 42416 |
| C_bursty | 200 | logistic_regression | test | 0.503 | 0.178 | 0.146 | 0.000 | 40928 |
| C_bursty | 200 | hist_gradient_boosting | train | 0.723 | 0.445 | 0.133 | 0.042 | 196944 |
| C_bursty | 200 | hist_gradient_boosting | val | 0.514 | 0.187 | 0.148 | 0.005 | 42416 |
| C_bursty | 200 | hist_gradient_boosting | test | 0.512 | 0.181 | 0.147 | 0.003 | 40928 |
| C_bursty | 500 | logistic_regression | train | 0.550 | 0.378 | 0.225 | 0.000 | 196944 |
| C_bursty | 500 | logistic_regression | val | 0.551 | 0.386 | 0.226 | 0.000 | 42416 |
| C_bursty | 500 | logistic_regression | test | 0.546 | 0.371 | 0.225 | 0.000 | 40928 |
| C_bursty | 500 | hist_gradient_boosting | train | 0.733 | 0.623 | 0.205 | 0.131 | 196944 |
| C_bursty | 500 | hist_gradient_boosting | val | 0.536 | 0.369 | 0.229 | 0.035 | 42416 |
| C_bursty | 500 | hist_gradient_boosting | test | 0.536 | 0.359 | 0.228 | 0.027 | 40928 |
| D_long_range_reuse [redundant w/ H=500] | 200 | logistic_regression | train | 1.000 | 1.000 | 0.000 | 1.000 | 4444160 |
| D_long_range_reuse [redundant w/ H=500] | 200 | logistic_regression | val | 1.000 | 1.000 | 0.000 | 1.000 | 952320 |
| D_long_range_reuse [redundant w/ H=500] | 200 | logistic_regression | test | 1.000 | 1.000 | 0.000 | 1.000 | 952320 |
| D_long_range_reuse [redundant w/ H=500] | 200 | hist_gradient_boosting | train | 1.000 | 1.000 | 0.000 | 1.000 | 4444160 |
| D_long_range_reuse [redundant w/ H=500] | 200 | hist_gradient_boosting | val | 1.000 | 1.000 | 0.000 | 1.000 | 952320 |
| D_long_range_reuse [redundant w/ H=500] | 200 | hist_gradient_boosting | test | 1.000 | 1.000 | 0.000 | 1.000 | 952320 |
| D_long_range_reuse [redundant w/ H=200] | 500 | logistic_regression | train | 1.000 | 1.000 | 0.000 | 1.000 | 4444160 |
| D_long_range_reuse [redundant w/ H=200] | 500 | logistic_regression | val | 1.000 | 1.000 | 0.000 | 1.000 | 952320 |
| D_long_range_reuse [redundant w/ H=200] | 500 | logistic_regression | test | 1.000 | 1.000 | 0.000 | 1.000 | 952320 |
| D_long_range_reuse [redundant w/ H=200] | 500 | hist_gradient_boosting | train | 1.000 | 1.000 | 0.000 | 1.000 | 4444160 |
| D_long_range_reuse [redundant w/ H=200] | 500 | hist_gradient_boosting | val | 1.000 | 1.000 | 0.000 | 1.000 | 952320 |
| D_long_range_reuse [redundant w/ H=200] | 500 | hist_gradient_boosting | test | 1.000 | 1.000 | 0.000 | 1.000 | 952320 |
| E_phase_changing | 25 | logistic_regression | train | 0.589 | 0.544 | 0.241 | 0.591 | 2289936 |
| E_phase_changing | 25 | logistic_regression | val | 0.590 | 0.545 | 0.241 | 0.593 | 495728 |
| E_phase_changing | 25 | logistic_regression | test | 0.588 | 0.543 | 0.242 | 0.593 | 493216 |
| E_phase_changing | 25 | hist_gradient_boosting | train | 0.607 | 0.571 | 0.234 | 0.663 | 2289936 |
| E_phase_changing | 25 | hist_gradient_boosting | val | 0.591 | 0.546 | 0.236 | 0.659 | 495728 |
| E_phase_changing | 25 | hist_gradient_boosting | test | 0.590 | 0.546 | 0.236 | 0.659 | 493216 |
| E_phase_changing | 50 | logistic_regression | train | 0.688 | 0.787 | 0.185 | 0.836 | 2289936 |
| E_phase_changing | 50 | logistic_regression | val | 0.689 | 0.789 | 0.184 | 0.838 | 495728 |
| E_phase_changing | 50 | logistic_regression | test | 0.686 | 0.785 | 0.185 | 0.836 | 493216 |
| E_phase_changing | 50 | hist_gradient_boosting | train | 0.700 | 0.799 | 0.175 | 0.846 | 2289936 |
| E_phase_changing | 50 | hist_gradient_boosting | val | 0.690 | 0.789 | 0.176 | 0.846 | 495728 |
| E_phase_changing | 50 | hist_gradient_boosting | test | 0.688 | 0.785 | 0.177 | 0.843 | 493216 |
| E_phase_changing | 100 | logistic_regression | train | 0.855 | 0.942 | 0.101 | 0.919 | 2289936 |
| E_phase_changing | 100 | logistic_regression | val | 0.856 | 0.943 | 0.100 | 0.920 | 495728 |
| E_phase_changing | 100 | logistic_regression | test | 0.857 | 0.944 | 0.100 | 0.919 | 493216 |
| E_phase_changing | 100 | hist_gradient_boosting | train | 0.862 | 0.947 | 0.097 | 0.921 | 2289936 |
| E_phase_changing | 100 | hist_gradient_boosting | val | 0.858 | 0.944 | 0.098 | 0.922 | 495728 |
| E_phase_changing | 100 | hist_gradient_boosting | test | 0.859 | 0.945 | 0.098 | 0.921 | 493216 |
| E_phase_changing [redundant w/ H=500] | 200 | logistic_regression | train | 0.913 | 0.977 | 0.083 | 0.930 | 2289936 |
| E_phase_changing [redundant w/ H=500] | 200 | logistic_regression | val | 0.914 | 0.977 | 0.083 | 0.931 | 495728 |
| E_phase_changing [redundant w/ H=500] | 200 | logistic_regression | test | 0.914 | 0.977 | 0.083 | 0.930 | 493216 |
| E_phase_changing [redundant w/ H=500] | 200 | hist_gradient_boosting | train | 0.917 | 0.978 | 0.081 | 0.933 | 2289936 |
| E_phase_changing [redundant w/ H=500] | 200 | hist_gradient_boosting | val | 0.913 | 0.977 | 0.082 | 0.933 | 495728 |
| E_phase_changing [redundant w/ H=500] | 200 | hist_gradient_boosting | test | 0.914 | 0.977 | 0.082 | 0.932 | 493216 |
| E_phase_changing [redundant w/ H=200] | 500 | logistic_regression | train | 0.914 | 0.977 | 0.083 | 0.930 | 2289936 |
| E_phase_changing [redundant w/ H=200] | 500 | logistic_regression | val | 0.914 | 0.977 | 0.083 | 0.931 | 495728 |
| E_phase_changing [redundant w/ H=200] | 500 | logistic_regression | test | 0.914 | 0.977 | 0.083 | 0.930 | 493216 |
| E_phase_changing [redundant w/ H=200] | 500 | hist_gradient_boosting | train | 0.918 | 0.978 | 0.081 | 0.933 | 2289936 |
| E_phase_changing [redundant w/ H=200] | 500 | hist_gradient_boosting | val | 0.914 | 0.977 | 0.082 | 0.933 | 495728 |
| E_phase_changing [redundant w/ H=200] | 500 | hist_gradient_boosting | test | 0.914 | 0.977 | 0.082 | 0.932 | 493216 |
| F_non_stationary | 25 | logistic_regression | train | 0.785 | 0.728 | 0.182 | 0.660 | 1832928 |
| F_non_stationary | 25 | logistic_regression | val | 0.787 | 0.732 | 0.180 | 0.659 | 390128 |
| F_non_stationary | 25 | logistic_regression | test | 0.787 | 0.730 | 0.181 | 0.661 | 392816 |
| F_non_stationary | 25 | hist_gradient_boosting | train | 0.813 | 0.782 | 0.167 | 0.672 | 1832928 |
| F_non_stationary | 25 | hist_gradient_boosting | val | 0.809 | 0.777 | 0.168 | 0.667 | 390128 |
| F_non_stationary | 25 | hist_gradient_boosting | test | 0.809 | 0.776 | 0.168 | 0.669 | 392816 |
| F_non_stationary | 50 | logistic_regression | train | 0.782 | 0.823 | 0.188 | 0.751 | 1832928 |
| F_non_stationary | 50 | logistic_regression | val | 0.785 | 0.824 | 0.187 | 0.751 | 390128 |
| F_non_stationary | 50 | logistic_regression | test | 0.782 | 0.824 | 0.188 | 0.752 | 392816 |
| F_non_stationary | 50 | hist_gradient_boosting | train | 0.813 | 0.864 | 0.173 | 0.750 | 1832928 |
| F_non_stationary | 50 | hist_gradient_boosting | val | 0.809 | 0.860 | 0.175 | 0.747 | 390128 |
| F_non_stationary | 50 | hist_gradient_boosting | test | 0.807 | 0.861 | 0.176 | 0.747 | 392816 |
| F_non_stationary | 100 | logistic_regression | train | 0.781 | 0.897 | 0.165 | 0.833 | 1832928 |
| F_non_stationary | 100 | logistic_regression | val | 0.782 | 0.896 | 0.166 | 0.831 | 390128 |
| F_non_stationary | 100 | logistic_regression | test | 0.777 | 0.895 | 0.166 | 0.830 | 392816 |
| F_non_stationary | 100 | hist_gradient_boosting | train | 0.817 | 0.924 | 0.152 | 0.841 | 1832928 |
| F_non_stationary | 100 | hist_gradient_boosting | val | 0.809 | 0.920 | 0.156 | 0.836 | 390128 |
| F_non_stationary | 100 | hist_gradient_boosting | test | 0.806 | 0.919 | 0.156 | 0.837 | 392816 |
| F_non_stationary | 200 | logistic_regression | train | 0.780 | 0.947 | 0.115 | 0.914 | 1832928 |
| F_non_stationary | 200 | logistic_regression | val | 0.779 | 0.947 | 0.116 | 0.912 | 390128 |
| F_non_stationary | 200 | logistic_regression | test | 0.776 | 0.947 | 0.114 | 0.916 | 392816 |
| F_non_stationary | 200 | hist_gradient_boosting | train | 0.831 | 0.965 | 0.104 | 0.920 | 1832928 |
| F_non_stationary | 200 | hist_gradient_boosting | val | 0.818 | 0.961 | 0.108 | 0.918 | 390128 |
| F_non_stationary | 200 | hist_gradient_boosting | test | 0.814 | 0.961 | 0.106 | 0.919 | 392816 |
| F_non_stationary | 500 | logistic_regression | train | 0.826 | 0.982 | 0.050 | 0.968 | 1832928 |
| F_non_stationary | 500 | logistic_regression | val | 0.828 | 0.982 | 0.050 | 0.967 | 390128 |
| F_non_stationary | 500 | logistic_regression | test | 0.824 | 0.982 | 0.049 | 0.968 | 392816 |
| F_non_stationary | 500 | hist_gradient_boosting | train | 0.900 | 0.992 | 0.043 | 0.971 | 1832928 |
| F_non_stationary | 500 | hist_gradient_boosting | val | 0.880 | 0.990 | 0.046 | 0.970 | 390128 |
| F_non_stationary | 500 | hist_gradient_boosting | test | 0.874 | 0.990 | 0.046 | 0.970 | 392816 |
