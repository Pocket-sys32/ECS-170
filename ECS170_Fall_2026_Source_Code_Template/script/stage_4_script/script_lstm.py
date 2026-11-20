'''
Concrete MethodRNN class for a specific learning of text
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from pathlib import Path
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Setting_PreSplit import Setting_PreSplit
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_1_code.Result_Saver import Result_Saver

def main():
    project_root = Path(__file__).resolve().parents[2]
    result_folder = project_root / "result" / "stage_4_result"
    result_folder.mkdir(parents=True, exist_ok=True)

    dataset_obj = Dataset_Loader(
        "IMDb",
        "Movie review sentiment classification",
        max_vocab_size=30000,
        max_length=300,
    )
    dataset_obj.dataset_source_folder_path = str(
        project_root / "data" / "stage_4_data"
        / "stage_4_data" / "text_classification"
    )

    method_obj = Method_RNN(
        "RNN",
        "Movie review classifier",
        vocab_size=30000,
        embedding_dim=50,
        hidden_size=128,
        max_epoch=5,
        learning_rate=1e-3,
        batch_size=64,
        learning_curve_path=str(result_folder / "rnn_learning_curve.png"),
    )

    method_obj.glove_path = (
            project_root / "data" / "stage_4_data" / "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
    )

    result_obj = Result_Saver("RNN results", "")
    result_obj.result_destination_folder_path = str(result_folder) + "/"
    result_obj.result_destination_file_name = "rnn_predictions"

    evaluate_obj = Evaluate_Metrics("Classification metrics", "")

    setting_obj = Setting_PreSplit("Pre-split text classification", "")
    setting_obj.prepare(
        dataset_obj, method_obj, result_obj, evaluate_obj
    )

    print(f"Starting training on {method_obj.device}...", flush=True)
    setting_obj.load_run_save_evaluate()


if __name__ == "__main__":
    main()