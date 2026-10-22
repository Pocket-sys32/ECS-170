from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Setting_PreSplit import Setting_PreSplit
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics  # Your new class
from local_code.stage_1_code.Result_Saver import Result_Saver

# Create objects
data_train = Dataset_Loader('train', '')
data_train.dataset_source_folder_path = '../../data/stage_2_data/stage_2_data/'
data_train.dataset_source_file_name = 'train.csv'

data_test = Dataset_Loader('test', '')
data_test.dataset_source_folder_path = '../../data/stage_2_data/stage_2_data/'
data_test.dataset_source_file_name = 'test.csv'

method_obj = Method_MLP('multi-layer perceptron', '')
# Modify as needed for better performance
method_obj.max_epoch = 1500
method_obj.learning_rate = 1e-3

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
result_obj.result_destination_file_name = 'prediction_result'

evaluate_obj = Evaluate_Metrics('metrics', '')

# Run
setting_obj = Setting_PreSplit('pre-split setting', '')
print('Starting Stage 2 MLP Training...')
setting_obj.prepare(data_train, data_test, method_obj, result_obj, evaluate_obj)
acc, _ = setting_obj.load_run_save_evaluate()