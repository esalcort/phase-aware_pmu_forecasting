import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.signal import medfilt

from src.input_parser import get_input_args
from src.create_dataset import get_raw_data, get_transformed_time_series, get_split_data_set, reshape_with_batch_size
from src.preprocess import generate_run_length
from src.evaluate import print_all_results
import src.Predictor as pred

def main():
    args = get_input_args(sys.argv[0])
    data = get_raw_data(args)
    timeseries = get_transformed_time_series(args, data)
    # print(timeseries.df)
    
    target = args.input_counters[0] # This is the final value to predict
    min_observation = timeseries.df["original"][target].min()
    max_observation = timeseries.df["original"][target].max()
    
    X_train, y_train, X_test, y_test = get_split_data_set(args, timeseries)
     
    X_train, y_train, _, _ = reshape_with_batch_size(X_train, y_train, X_test, y_test, args.batch_size)
    train_data = pred.PredictorInputs(args, X_train, y_train,)
    test_data = pred.PredictorInputs(args, X_test, y_test, train_data.in_scaler, train_data.out_scaler, batch_padding=True)


    predictor = pred.SerialPredictor(args, train_data.X)
    predictions = predictor.train_predict(args, train_data, test_data)

    test_data.add_predictions(args, predictions)

    predictions_df = timeseries.invert_predicted_transforms(test_data.y_hat)

    # print(predictions_df)
    print_all_results(args, timeseries, predictions_df, test_data.y_hat, y_test, target)

if __name__ == "__main__":
    main()