import os
import argparse
import torch
from multiprocessing import Pool
from tabular_prediction.methods import lasso_predict
from tabular_prediction.metrics import accuracy_metric, balanced_accuracy_metric, cross_entropy_metric, auc_metric
import pandas as pd
import numpy as np
from read_data import get_datasets
import datetime
from handle_results import prepare_results_file, write_results
    
def run_evaluation(split):
    max_time = [1, 5, 10, 30, 60, 120, 300, 600, 3600]

    data_dir, datasets = get_datasets(split)
    
    result_file = os.path.abspath(f"../results/lasso-classification-{split}.csv")
    
    previous_results = prepare_results_file(result_file, max_time)
    if previous_results is None:
        exit()

    with open(result_file, "a") as f:
        for i, dataset in enumerate(datasets):
            data = torch.load(os.path.join(data_dir, dataset), map_location='cpu') # for GPUs - should this not be GPU?
            x_train, y_train, x_test, y_test = data["data"]
                        
            total_num_of_samples = (x_train.shape[0] + x_test.shape[0])
            if total_num_of_samples > 625:
                print(f"Skipping {dataset} total_num_of_samples={total_num_of_samples}")
                continue
            
            if dataset in previous_results:
                assert previous_results.loc[dataset] == len(max_time)
                continue
            
            cat_features = torch.where(data["cat_features"])[0].to(torch.int32)

            test_y, summary, _ = lasso_predict(x_train, y_train, x_test, y_test, cat_features=cat_features, metric_used=cross_entropy_metric, max_time=max_time)
            write_results(test_y, summary, max_time, dataset, file_handler=f)

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, help='split number - must be 1 to 6', default=1)
parser.add_argument('-m', '--multi_processing', action='store_true', help='whether to use python multiprocessing - this tries to spawn all splits from one python process')

args = parser.parse_args()

if args.multi_processing:
    print("Running with multiprocessing!")
    with Pool(processes=6) as p:
        print(p.map(run_evaluation, range(1, 7)))
else:
    split = args.split
    assert split in range(1,7)
    
    print(f"starting split {split}")
    run_evaluation(split=split)
    print(f"completed split {split}")
