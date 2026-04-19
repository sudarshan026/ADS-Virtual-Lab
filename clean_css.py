import os
import re

files = [
    r'D:\adsvl\ADS-exp-1-virtual-lab-main\app.py',
    r'D:\adsvl\ADS-VL-main exp 2\app.py',
    r'D:\adsvl\app exp 3.py',
    r'D:\adsvl\ADS_virtual_lab-main exp 4\virtual-lab-ui\app.py',
    r'D:\adsvl\ADS_VirtualLab_SMOTE-main exp 5\app.py',
    r'D:\adsvl\adsca exp 6.py',
    r'D:\adsvl\ADS_Virtual_Lab-main exp 7\exp7.py',
    r'D:\adsvl\VL-DS-main exp8\app.py',
    r'D:\adsvl\Exp_9_ADSVirtualLab-main\app.py',
    r'D:\adsvl\experiments\exp1_statistics.py',
    r'D:\adsvl\experiments\exp2_model_evaluation.py',
    r'D:\adsvl\experiments\exp3_visualization.py',
    r'D:\adsvl\experiments\exp4_data_cleaning.py',
    r'D:\adsvl\experiments\exp5_smote.py',
    r'D:\adsvl\experiments\exp6_outlier.py',
    r'D:\adsvl\experiments\exp7_timeseries.py',
    r'D:\adsvl\experiments\exp8_lifecycle.py',
    r'D:\adsvl\experiments\exp9_automl.py'
]

# We will remove  blocks to ensure global theme rules.
for fpath in files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove literal 
        new_content = re.sub(r'', '', content, flags=re.DOTALL)
        
        if content != new_content:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'Cleaned {os.path.basename(fpath)}')
    except Exception as e:
        print(f'Warning reading {fpath}: {e}')
