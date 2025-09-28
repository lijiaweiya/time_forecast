import os
import csv
import subprocess

# 路径配置

CSV_PATH = os.path.join( 'test_results', 'analysis_acc_gt_0.5.csv')
RUN_PY =  'run.py'

# 固定参数
params = [
    '--save_mode','True', # 用于模式保存，默认为False
    '--scale', 'True',
    '--err', 'False',
    '--task_name', 'long_term_forecast',
    '--train_epochs', '100',
    '--is_training', '1',
    '--root_path', './datasets/ETT-small/',
    '--data_path', 'ETTm1.csv',
    '--model_id', 'ETTm1_96_96',
    '--model', 'WSB_2',
    '--data', 'ETTm1',
    '--features', 'M',
    '--seq_len', '96',
    '--label_len', '0',
    '--pred_len', '96',
    '--e_layers', '5',
    '--enc_in', '7',
    '--c_out', '7',
    '--des', 'Exp',
    '--itr', '1',
    '--d_model', '16',
    '--d_ff', '32',
    '--batch_size', '16',
    '--learning_rate', '0.001',
    '--down_sampling_layers', '3',
    '--down_sampling_method', 'avg',
    '--down_sampling_window', '2',
]

def get_best_per_lie(csv_path):
    best = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            info = eval(row['info']) if isinstance(row['info'], str) else row['info']
            lie = info.get('lie')
            cluster = info.get('cluster')
            best_f1 = float(row['best_f1'])
            if lie not in best or best_f1 > best[lie]['best_f1']:
                best[lie] = {
                    'll': lie,
                    'cluster': cluster,
                    'best_f1': best_f1,
                    'accuracy': float(row['accuracy'])
                }
    return best

def main():
    best = get_best_per_lie(CSV_PATH)
    print('ll,cluster,best_f1,accuracy')
    for lie, v in best.items():
        print(f"{v['ll']},{v['cluster']},{v['best_f1']},{v['accuracy']}")
    print('\n生成如下训练命令：')
    for lie, v in best.items():
        cmd = ['python', RUN_PY, '--ll', str(v['ll']), '--cluster', str(v['cluster'])] + params
        print(' '.join(cmd))
        # 如需自动执行训练，取消下行注释
        subprocess.run(cmd)

if __name__ == '__main__':
    main()

