# HERBench repeated splits (supplementary bank)

Splits 1-30; split 0 is the reference assignment.

     model           method  mean_gain  min_gain  harm  splits
internvl14 all_view_support  13.931252 11.252919     0      30
internvl14     base_support  11.395312  9.019468     0      30
internvl14          checked  10.836439  2.201942     0      30
internvl14  development_hgb   9.317193  6.469159     0      30
internvl14         original   0.000000  0.000000     0      30
internvl14         pilot_lr  12.126899  2.201942     0      30
    llava8 all_view_support  12.105328  7.528959     0      30
    llava8     base_support   8.647305  4.134100     0      30
    llava8          checked  10.829747  5.387696     0      30
    llava8  development_hgb   7.913680  2.147752     0      30
    llava8         original   0.000000  0.000000     0      30
    llava8         pilot_lr  10.987065  6.016383     0      30
     qwen8 all_view_support  19.762123 15.554383     0      30
     qwen8     base_support  16.894656 12.248029     0      30
     qwen8          checked  19.326606 14.164156     0      30
     qwen8  development_hgb  16.559787 10.584641     0      30
     qwen8         original   0.000000  0.000000     0      30
     qwen8         pilot_lr  19.326606 14.164156     0      30

## Mean across three models

                       mean        min
method                                
all_view_support  15.266235  12.972087
base_support      12.312424  10.377739
checked           13.664264  10.539661
development_hgb   11.263553   8.320880
original           0.000000   0.000000
pilot_lr          14.146857  10.539661

## Checked minus label-free support (pp, splits 1-30)

- vs base_support: mean 1.35, positive in 25/30, min -1.80
- vs all_view_support: mean -1.60, positive in 2/30, min -4.58

## Selection counts

selected    source_trace_hgb  target_logistic
model                                        
internvl14                 9               21
llava8                     1               29
qwen8                      0               30

## Reference split 0

method      all_view_support  base_support    checked  development_hgb  original   pilot_lr
model                                                                                      
internvl14         11.907302      8.632047  11.410400         8.578838       0.0  11.410400
llava8             12.884135      9.038757  11.219188        10.142615       0.0  11.219188
qwen8              16.100400     11.543326  14.401959        14.530904       0.0  14.401959

## Accuracy on reference test split (new bank)

               accuracy
internvl14     0.472482
qwen8          0.526480
llava8         0.492212
majority_vote  0.523364
