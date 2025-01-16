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
    
def run_evaluation(split):
    max_time = [1, 5, 10, 30, 60, 120, 300, 600, 3600]

    data_dir, datasets = get_datasets(split)
    
    result_file = os.path.abspath(f"../results/lasso-classification-{split}.csv")
    
    results_schema = ["dataset_name", "acc", "bacc", "ce", "auc", "stop_time", "max_time"]
    
    if os.path.exists(result_file):
        print("Found existing results record")
        existing_results_df = pd.read_csv(result_file)
        
        # TODO: explore more code that will salvage old results!

        if (len(existing_results_df.columns) != len(results_schema)) or (existing_results_df.columns != results_schema).any():
            old_results_schema = ["dataset", "acc", "bacc", "ce", "auc", "time"]
            if (len(existing_results_df.columns) != len(old_results_schema)) or (existing_results_df.columns != old_results_schema).any():
                raise UnknownResultsSchema(f"Previous results file, has unkown schema: {existing_results_df.columns}")
            else:
                print("Found results are in old schema - attempting to correct")
                
                old_results_new_filename = result_file.replace(".csv", f"_old_schema_{datetime.datetime.utcnow()}.csv")
                existing_results_df.to_csv(old_results_new_filename, index=False)
                print(f"wrote a copy of the old result to: {old_results_new_filename}")
                
                existing_results_df = existing_results_df[existing_results_df.dataset.str.endswith(".pt")]
                
                existing_results_df.rename(columns={"dataset": "dataset_name", "time": "stop_time"}, inplace=True)
                
                num_of_rows_per_dataset = existing_results_df.groupby("dataset_name")["stop_time"].count()
                bad_datasets = num_of_rows_per_dataset[num_of_rows_per_dataset != len(max_time)]
                assert (num_of_rows_per_dataset == len(max_time)).all(), f"Error! some datasets did not have the right number of rows: {bad_datasets.head()}. Might be too hairy to salvage results from this file"
                
                existing_results_df["max_time"] = existing_results_df.groupby("dataset_name")["stop_time"].cumcount()
                existing_results_df["max_time"] = existing_results_df["max_time"].map(pd.Series(max_time))
                
                existing_results_df.to_csv(result_file, index=False)
                print("Revamped old results file to conform to new schema - please run again to continue generating results")
                exit()
        
        # we allow for duplicate run records - don't want to hampe the parallelization!
        previous_results = existing_results_df.groupby("dataset_name")["max_time"].nunique()
        print(previous_results.head())

    else:
        # initialize the results file
        with open(result_file, "a") as f:
            f.write(','.join(results_schema))
            f.write('\n')
            f.flush()
            print("Initialized results file - please run again to kick off runs generating results")
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
            
            assert len(summary) == len(max_time), "ALERT! somwhoe the number of summaries is different from the number of max_times we supplied"
            
            for stop_time, _max_time in zip(summary, max_time):
                pred = summary[stop_time]['pred']
                run_time = summary[stop_time]['tune_time'] + summary[stop_time]['train_time'] + summary[stop_time]['predict_time']
                f.write(','.join(
                    [dataset] + [f'{val:5.4f}' for val in 
                                 [accuracy_metric(test_y, pred), 
                                  balanced_accuracy_metric(test_y, pred), 
                                  cross_entropy_metric(test_y, pred), 
                                  auc_metric(test_y, pred), 
                                  run_time]
                                 ]
                    + [str(_max_time)]))
                f.write('\n')
                f.flush()
                

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
