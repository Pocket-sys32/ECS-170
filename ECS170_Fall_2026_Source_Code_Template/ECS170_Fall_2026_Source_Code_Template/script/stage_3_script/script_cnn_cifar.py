from local_code.stage_3_code.Dataset_Loader_CIFAR import Dataset_Loader_CIFAR
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Setting_CNN_CIFAR import Setting_CNN_CIFAR
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_1_code.Result_Saver import Result_Saver
import numpy as np
import torch
import matplotlib.pyplot as plt # for generating results plot

np.random.seed(2)
torch.manual_seed(2)

data_obj = Dataset_Loader_CIFAR('CIFAR-10 Dataset', 'CIFAR Object Dataset Loader')
data_obj.dataset_source_folder_path = '../../data/stage_3_data'
data_obj.dataset_source_file_name = 'CIFAR'  # Exact dataset file name without extension

method_obj = Method_CNN_CIFAR('CNN CIFAR', 'Convolutional Neural Network for CIFAR-10')
method_obj.max_epoch = 20

# save results to a file
result_obj = Result_Saver('Saver', '')
result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
result_obj.result_destination_file_name = 'prediction_result'

evaluate_obj = Evaluate_Metrics('CIFAR metrics', 'Multi-class Classification Evaluation')
evaluate_obj.log_path = "../../result/stage_3_result/metrics_log.csv"
evaluate_obj.run_label = 'cnn_cifar_FINAL_2conv_32_64_k3_20epochs'

setting_obj = Setting_CNN_CIFAR('CIFAR Setting', 'Setting script for CIFAR')
#setting_obj.prepare(data_obj, method_obj, sResult=None, sEvaluate=evaluate_obj)

print('Starting Stage 3 CIFAR CNN...')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
acc, _ = setting_obj.load_run_save_evaluate()
#print('Accuracy:', acc)
plt.plot(method_obj.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title('CIFAR-10 CNN training loss')
plt.savefig('../../result/stage_3_result/cifar_loss_curve.png')
#plt.show()