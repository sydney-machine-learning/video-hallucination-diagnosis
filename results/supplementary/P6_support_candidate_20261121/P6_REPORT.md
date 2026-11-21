# Support as a third candidate (post hoc, splits 1-30)

## HERBench

            rule2_gain  rule3_gain  support_gain  pilot_lr_gain
model                                                          
internvl14       10.84       13.21         13.93          12.13
llava8           10.83       11.80         12.11          10.99
qwen8            19.33       19.50         19.76          19.33

Mean across models: rule2_gain 13.66, rule3_gain 14.84, support_gain 15.27, pilot_lr_gain 14.15

Harm (evaluations < 0 of 90): rule2_gain 0, rule3_gain 0, support_gain 0, pilot_lr_gain 0
Rule3 minus support, per split: mean -0.43, rule3 ahead in 0/30
Rule3 minus rule2, per split: mean 1.17, min split mean rule3 11.49

Rule3 selections:
rule3_selected  all_view_support  source_trace_hgb  target_logistic
model                                                              
internvl14                    26                 1                3
llava8                        27                 0                3
qwen8                         25                 0                5

Rule2 selections:
rule2_selected  source_trace_hgb  target_logistic
model                                            
internvl14                     9               21
llava8                         1               29
qwen8                          0               30

## Perception

            rule2_gain  rule3_gain  support_gain  pilot_lr_gain
model                                                          
internvl14        0.26        0.16         -0.43           5.75
llava8           26.61       26.31         26.00          28.14
qwen8            17.75       16.87          9.96          18.69

Mean across models: rule2_gain 14.87, rule3_gain 14.45, support_gain 11.84, pilot_lr_gain 17.53

Harm (evaluations < 0 of 90): rule2_gain 0, rule3_gain 0, support_gain 17, pilot_lr_gain 0
Rule3 minus support, per split: mean 2.60, rule3 ahead in 27/30
Rule3 minus rule2, per split: mean -0.42, min split mean rule3 9.06

Rule3 selections:
rule3_selected  all_view_support  raw_fallback  source_trace_hgb  target_logistic
model                                                                            
internvl14                     0            28                 0                2
llava8                        15             0                 2               13
qwen8                          0             2                 1               27

Rule2 selections:
rule2_selected  raw_fallback  source_trace_hgb  target_logistic
model                                                          
internvl14                27                 0                3
llava8                     0                13               17
qwen8                      1                 1               28

