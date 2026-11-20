import sys
from pathlib import Path

from local_code.stage_1_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import (
    Method_CNN_CIFAR,
    Method_CNN_MNIST,
    Method_CNN_ORL,
)
from local_code.stage_3_code.Setting_PreSplit import Setting_PreSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / 'data' / 'stage_3_data'
RESULT_FOLDER = PROJECT_ROOT / 'result' / 'stage_3_result'

DATASET_CONFIG = {
    'MNIST': (Method_CNN_MNIST, 'handwritten digit classification'),
    'ORL': (Method_CNN_ORL, 'face recognition'),
    'CIFAR': (Method_CNN_CIFAR, 'colored object classification'),
}


def main():
    dataset_name = sys.argv[1].upper() if len(sys.argv) > 1 else 'MNIST'
    if dataset_name not in DATASET_CONFIG:
        raise SystemExit('Choose a dataset: MNIST, ORL, or CIFAR')

    method_class, description = DATASET_CONFIG[dataset_name]
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

    dataset_obj = Dataset_Loader(dataset_name, description)
    dataset_obj.dataset_source_folder_path = str(DATA_FOLDER) + '\\'
    dataset_obj.dataset_source_file_name = dataset_name

    method_obj = method_class(
        f'CNN {dataset_name}',
        description,
        max_epoch=15,
        learning_rate=1e-3,
        batch_size=128,
        learning_curve_path=str(RESULT_FOLDER / f'{dataset_name.lower()}_learning_curve.png'),
    )

    result_obj = Result_Saver('stage 3 results', '')
    result_obj.result_destination_folder_path = str(RESULT_FOLDER) + '\\'
    result_obj.result_destination_file_name = f'{dataset_name}_cnn_predictions'

    evaluate_obj = Evaluate_Metrics('metrics', '')
    setting_obj = Setting_PreSplit('pre-split image setting', '')
    setting_obj.prepare(dataset_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()


if __name__ == '__main__':
    main()
