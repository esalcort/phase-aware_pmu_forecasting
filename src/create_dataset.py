import src.traces as traces
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
