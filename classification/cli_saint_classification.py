import os
import argparse
import torch
from multiprocessing import Pool
from tabular_prediction.methods import saint_predict
from tabular_prediction.metrics import accuracy_metric, balanced_accuracy_metric, cross_entropy_metric, auc_metric

# all datasets

from read_data import get_datasets
from handle_results import prepare_results_file, write_results

def run_evaluation(split, gpu_id=0, parallelize_datasets=False):
    max_time = [1, 5, 10, 30, 60, 120, 300, 600, 3600]

    data_dir, datasets = get_datasets(split)

    result_file = f"../results/saint-classification-{split}.csv"

    previous_results = prepare_results_file(result_file, max_time)
    if previous_results is None:
        exit()
        
    with open(result_file, "a") as f:
        for i, dataset in enumerate(datasets):
            if parallelize_datasets:
                run_id="_".join([dataset.split(".")[0], str(split)])
            else:
                run_id = str(split)
                
            try:
                data = torch.load(os.path.join(data_dir, dataset), map_location='cpu')
            except UnpicklingError:
                href_str = "https://stackoverflow.com/questions/33049688/what-causes-the-error-pickle-unpicklingerror-invalid-load-key"
                print(f"Oy Vey! might have to recreate {dataset} - it may have been gzipped at some point, and that might have ruined it. \nSee {href_str}")
                continue
            x_train, y_train, x_test, y_test = data["data"]
            
            total_num_of_samples = (x_train.shape[0] + x_test.shape[0])
            if total_num_of_samples > 625:
                print(f"Skipping {dataset} total_num_of_samples={total_num_of_samples}")
                continue
            
            if dataset in previous_results:
                assert previous_results.loc[dataset] == len(max_time)
                continue
            
            cat_features = torch.where(data["cat_features"])[0]

            test_y, summary, _ = saint_predict(
                x_train, y_train, x_test, y_test, cat_features=cat_features, 
                metric_used=cross_entropy_metric, max_time=max_time, gpu_id=gpu_id, 
                run_id=run_id,
                )
            write_results(test_y, summary, max_time, dataset, file_handler=f)
                

parser = argparse.ArgumentParser()
parser.add_argument('split', type=int, help='split number - must be 1 to 6')
parser.add_argument('gpu', type=int, help='which gpu to use - 0, 1, ...')

args = parser.parse_args()

split = args.split
assert split in range(1,7)

print(f"starting split {split}")
run_evaluation(split=split, gpu_id=args.gpu)
print(f"completed split {split}")

