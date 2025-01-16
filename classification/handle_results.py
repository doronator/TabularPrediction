import pandas as pd
import os 
import datetime
from tabular_prediction.metrics import accuracy_metric, balanced_accuracy_metric, cross_entropy_metric, auc_metric

def prepare_results_file(result_file, max_time):
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
        return previous_results

    else:
        # initialize the results file
        with open(result_file, "a") as f:
            f.write(','.join(results_schema))
            f.write('\n')
            f.flush()
            print("Initialized results file - please run again to kick off runs generating results")
            exit()
            

def write_results(test_y, summary, max_time, dataset, file_handler):
    assert len(summary) == len(max_time), "ALERT! somehow the number of summaries is different from the number of max_times we supplied"
    
    for stop_time, _max_time in zip(summary, max_time):
        pred = summary[stop_time]['pred']
        run_time = summary[stop_time]['tune_time'] + summary[stop_time]['train_time'] + summary[stop_time]['predict_time']
        file_handler.write(','.join(
            [dataset] + [f'{val:5.4f}' for val in 
                            [accuracy_metric(test_y, pred), 
                            balanced_accuracy_metric(test_y, pred), 
                            cross_entropy_metric(test_y, pred), 
                            auc_metric(test_y, pred), 
                            run_time]
                            ]
            + [str(_max_time)]))
        file_handler.write('\n')
        file_handler.flush()