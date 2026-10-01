Table 14 -- RQ6: logistic regression vs. histogram gradient boosting, train-vs-test ROC-AUC and the gap between them (a larger gap indicates more overfitting), per family/horizon. 'Selected?' marks which model Section 13.3's post-hoc validation-ROC-AUC selection chose as the Learned Utility Policy for that (family, horizon) -- the Protocol A/B operational tables (9, 10) always report the SELECTED model's results, not necessarily HGB. Source: models.<name>.{train,test}_metrics.roc_auc in each protocol_a_*.json checkpoint.

| Family | H | Model | Train ROC-AUC | Test ROC-AUC | Gap | Selected? |
| --- | --- | --- | --- | --- | --- | --- |
| A_uniform_random | 25 | logistic_regression | 0.505 | 0.500 | 0.005 | yes |
| A_uniform_random | 25 | hist_gradient_boosting | 0.571 | 0.502 | 0.069 | no |
| A_uniform_random | 50 | logistic_regression | 0.510 | 0.511 | -0.001 | yes |
| A_uniform_random | 50 | hist_gradient_boosting | 0.574 | 0.513 | 0.061 | no |
| A_uniform_random | 100 | logistic_regression | 0.538 | 0.537 | 0.001 | no |
| A_uniform_random | 100 | hist_gradient_boosting | 0.598 | 0.540 | 0.058 | yes |
| A_uniform_random | 200 | logistic_regression | 0.689 | 0.686 | 0.003 | no |
| A_uniform_random | 200 | hist_gradient_boosting | 0.745 | 0.689 | 0.056 | yes |
| A_uniform_random | 500 | logistic_regression | 0.970 | 0.969 | 0.001 | yes |
| A_uniform_random | 500 | hist_gradient_boosting | 0.977 | 0.968 | 0.009 | no |
| B_temporal_locality | 25 | logistic_regression | 0.845 | 0.844 | 0.001 | no |
| B_temporal_locality | 25 | hist_gradient_boosting | 0.930 | 0.850 | 0.080 | yes |
| B_temporal_locality | 50 | logistic_regression | 0.847 | 0.835 | 0.012 | yes |
| B_temporal_locality | 50 | hist_gradient_boosting | 0.927 | 0.836 | 0.090 | no |
| B_temporal_locality | 100 | logistic_regression | 0.849 | 0.832 | 0.017 | no |
| B_temporal_locality | 100 | hist_gradient_boosting | 0.930 | 0.830 | 0.100 | yes |
| B_temporal_locality | 200 | logistic_regression | 0.839 | 0.823 | 0.017 | yes |
| B_temporal_locality | 200 | hist_gradient_boosting | 0.933 | 0.817 | 0.117 | no |
| B_temporal_locality | 500 | logistic_regression | 0.830 | 0.808 | 0.021 | yes |
| B_temporal_locality | 500 | hist_gradient_boosting | 0.953 | 0.797 | 0.156 | no |
| C_bursty | 25 | logistic_regression | 0.518 | 0.478 | 0.040 | yes |
| C_bursty | 25 | hist_gradient_boosting | 0.773 | 0.502 | 0.271 | no |
| C_bursty | 50 | logistic_regression | 0.515 | 0.491 | 0.024 | no |
| C_bursty | 50 | hist_gradient_boosting | 0.745 | 0.518 | 0.227 | yes |
| C_bursty | 100 | logistic_regression | 0.510 | 0.497 | 0.013 | yes |
| C_bursty | 100 | hist_gradient_boosting | 0.722 | 0.518 | 0.204 | no |
| C_bursty | 200 | logistic_regression | 0.513 | 0.503 | 0.010 | no |
| C_bursty | 200 | hist_gradient_boosting | 0.723 | 0.512 | 0.211 | yes |
| C_bursty | 500 | logistic_regression | 0.550 | 0.546 | 0.005 | yes |
| C_bursty | 500 | hist_gradient_boosting | 0.733 | 0.536 | 0.197 | no |
| D_long_range_reuse [redundant w/ H=500] | 200 | logistic_regression | 1.000 | 1.000 | 0.000 | yes |
| D_long_range_reuse [redundant w/ H=500] | 200 | hist_gradient_boosting | 1.000 | 1.000 | 0.000 | no |
| D_long_range_reuse [redundant w/ H=200] | 500 | logistic_regression | 1.000 | 1.000 | 0.000 | yes |
| D_long_range_reuse [redundant w/ H=200] | 500 | hist_gradient_boosting | 1.000 | 1.000 | 0.000 | no |
| E_phase_changing | 25 | logistic_regression | 0.589 | 0.588 | 0.001 | no |
| E_phase_changing | 25 | hist_gradient_boosting | 0.607 | 0.590 | 0.017 | yes |
| E_phase_changing | 50 | logistic_regression | 0.688 | 0.686 | 0.002 | no |
| E_phase_changing | 50 | hist_gradient_boosting | 0.700 | 0.688 | 0.011 | yes |
| E_phase_changing | 100 | logistic_regression | 0.855 | 0.857 | -0.003 | no |
| E_phase_changing | 100 | hist_gradient_boosting | 0.862 | 0.859 | 0.002 | yes |
| E_phase_changing [redundant w/ H=500] | 200 | logistic_regression | 0.913 | 0.914 | -0.001 | yes |
| E_phase_changing [redundant w/ H=500] | 200 | hist_gradient_boosting | 0.917 | 0.914 | 0.003 | no |
| E_phase_changing [redundant w/ H=200] | 500 | logistic_regression | 0.914 | 0.914 | -0.001 | yes |
| E_phase_changing [redundant w/ H=200] | 500 | hist_gradient_boosting | 0.918 | 0.914 | 0.004 | no |
| F_non_stationary | 25 | logistic_regression | 0.785 | 0.787 | -0.002 | no |
| F_non_stationary | 25 | hist_gradient_boosting | 0.813 | 0.809 | 0.003 | yes |
| F_non_stationary | 50 | logistic_regression | 0.782 | 0.782 | -0.000 | no |
| F_non_stationary | 50 | hist_gradient_boosting | 0.813 | 0.807 | 0.005 | yes |
| F_non_stationary | 100 | logistic_regression | 0.781 | 0.777 | 0.004 | no |
| F_non_stationary | 100 | hist_gradient_boosting | 0.817 | 0.806 | 0.011 | yes |
| F_non_stationary | 200 | logistic_regression | 0.780 | 0.776 | 0.005 | no |
| F_non_stationary | 200 | hist_gradient_boosting | 0.831 | 0.814 | 0.016 | yes |
| F_non_stationary | 500 | logistic_regression | 0.826 | 0.824 | 0.002 | no |
| F_non_stationary | 500 | hist_gradient_boosting | 0.900 | 0.874 | 0.026 | yes |
