import os
import json
import csv

# 结果目录
RESULT_DIR = os.path.join(
    os.path.dirname(__file__),
    'test_results/long_term_forecast_ETTm1_96_96_WSB_2_ETTm1_ftM_sl96_ll0_pl96_dm16_nh8_el5_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/'
)

# 输出文件
OUTPUT_CSV = os.path.join('test_results/analysis_acc_gt_0.6.csv')

results = []

for fname in os.listdir(RESULT_DIR):
    if not fname.endswith('.json'):
        continue
    fpath = os.path.join(RESULT_DIR, fname)
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        info = data.get('info', {})
        report = data.get('report', {})
        acc = report.get('accuracy', 0)
        if acc > 0.6:
            # 找到f1-score最大的类别（排除聚合项）
            best_cls = None
            best_f1 = -1
            for k, v in report.items():
                if k in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                if isinstance(v, dict) and 'f1-score' in v:
                    if v['f1-score'] > best_f1:
                        best_f1 = v['f1-score']
                        best_cls = k
            results.append({
                'file': fname,
                'info': info,
                'accuracy': acc,
                'best_class': best_cls,
                'best_f1': best_f1
            })
    except Exception as e:
        print(f'Error processing {fname}: {e}')

# 输出到csv
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'info', 'accuracy', 'best_class', 'best_f1'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f'Done! {len(results)} results with accuracy > 0.6. Output: {OUTPUT_CSV}')

