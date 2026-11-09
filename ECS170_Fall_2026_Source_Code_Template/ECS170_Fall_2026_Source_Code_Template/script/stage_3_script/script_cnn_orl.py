from local_code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader_ORL
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Setting_ORL import Setting_ORL
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_1_code.Result_Saver import Result_Saver
import numpy as np
import torch
import matplotlib.pyplot as plt


# fix random seeds so results are reproducible
np.random.seed(2)
torch.manual_seed(2)

# objects
data_obj = Dataset_Loader_ORL('ORL', '')
data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
data_obj.dataset_source_file_name = 'ORL'

# modify for performance if needed

method_obj = Method_CNN_ORL('CNN for ORL', '')
method_obj.max_epoch = 60
method_obj.learning_rate = 1e-3

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_ORL_'
result_obj.result_destination_file_name = 'prediction_result'

evaluate_obj = Evaluate_Metrics('metrics', '')

setting_obj = Setting_ORL('pre-split setting', '')

# run
print('Starting Stage 3 ORL CNN Training...')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
acc, _ = setting_obj.load_run_save_evaluate()

# convergence plot
plt.plot(method_obj.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Training Loss (Cross-Entropy)')
plt.title('ORL CNN Training Loss')
plt.yscale('log')
plt.savefig('../../result/stage_3_result/loss_curve_orl.png', dpi=200)
plt.show()