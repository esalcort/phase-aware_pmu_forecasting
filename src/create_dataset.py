import src.traces as traces
import src.TimeSeries as ts
import pandas as pd

def get_raw_data(args):
	bm = args.benchmark
	df = traces.get_raw_data(bm, dataset=args.dataset)
	data = pd.concat([df, traces.get_processed_data(df)], axis=1)
	if data.columns.nlevels == 1:
		data = data.loc[:, args.input_counters]
	else:
		data = data.swaplevel(0, axis=1).loc[:, args.input_counters].swaplevel(0, axis=1)
	if args.end_drop_count != 0:
		data = data.iloc[args.start_drop_count:-args.end_drop_count,:]
	else:
		data = data.iloc[args.start_drop_count:,:]
	return data

def get_transformed_time_series(args, data):
	timeseries = ts.TimeSeriesTransforms(data)
	timeseries.set_transforms(args)
	return timeseries

def get_split_data_set(args, timeseries):
	timeseries.set_train_test_split(args)
	return timeseries.X_train, timeseries.y_train, timeseries.X_test, timeseries.y_test


def get_drop_samples(data_shape, batch_size):
	assert data_shape >= batch_size, "Batch size is too large"
	batch_count = int(data_shape / batch_size)
	drop_samples = 0
	if batch_size > 1:
		drop_samples = data_shape - (batch_count * batch_size)
	return drop_samples

def reshape_with_batch_size(X, y, vX, vy, batch_size):
	train_drop_samples = get_drop_samples(X.shape[0], batch_size)
	test_drop_samples = get_drop_samples(vX.shape[0], batch_size)
	if train_drop_samples > 0:
		X = X.iloc[train_drop_samples:]
		y = y.iloc[train_drop_samples:]
	if test_drop_samples > 0:
		vX = vX.iloc[:-test_drop_samples]
		vy = vy.iloc[:-test_drop_samples]
	return X, y, vX, vy
