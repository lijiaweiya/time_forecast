# time_forecast

## 原项目地址
[Time-Series-Library](https://github.com/thuml/Time-Series-Library "时间序列库")




## 远程文件路径
```bash
cd /home/gf-shu/wsb/time_forecast
conda activate time
```

## analyze_results.py 说明
该脚本用于自动分析 test_results 目录下的实验结果，主要关注 info 和 report 字段中的性能指标。脚本会筛选出 accuracy 大于 0.6（或 0.5，可根据实际需求修改）的实验结果，并找到每个实验中 f1-score 最好的类别。最终结果会保存为 test_results/analysis_acc_gt_0.6.csv（或 analysis_acc_gt_0.5.csv），便于后续分析和自动化处理。

## gen_best_train_cmds.py 说明
该脚本根据 analysis_acc_gt_0.5.csv 的分析结果，为每个 lie 找到 f1-score 最好的聚类类别，并自动生成训练命令。每条命令会指定最佳的 ll（列）和 cluster（聚类数），用于后续模型训练和预测。可直接打印命令或自动执行训练。

## compare_modes.py 说明
该脚本用于对比 test_results/mode 目录下的模式文件。会对指定的两次训练的结果对进行 shape 和内容一致性检查，输出 shape 是否一致及内容是否完全一致的信息，用于验证模式提取和保存的正确性。
