from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Setting_MNIST import Setting_MNIST
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_1_code.Result_Saver import Result_Saver
import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(2)
torch.manual_seed(2)

data_obj = Dataset_Loader_MNIST('mnist', '')
data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
data_obj.dataset_source_file_name = 'MNIST'

method_obj = Method_CNN_MNIST('cnn mnist', '')
method_obj.max_epoch = 10

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_MNIST_'
result_obj.result_destination_file_name = 'prediction_result'

evaluate_obj = Evaluate_Metrics('metrics', '')
evaluate_obj.log_path = '../../result/stage_3_result/metrics_log.csv'
evaluate_obj.run_label = 'cnn_mnist_FINAL_2conv_16_32_k5_10epochs'

setting_obj = Setting_MNIST('mnist setting', '')

print('Starting Stage 3 MNIST CNN...')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
acc, _ = setting_obj.load_run_save_evaluate()
print('Accuracy:', acc)
plt.plot(method_obj.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title('MNIST CNN training loss')
plt.savefig('../../result/stage_3_result/mnist_loss_curve.png')
plt.show()