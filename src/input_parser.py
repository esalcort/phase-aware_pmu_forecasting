import argparse

def set_experiment_args(parser):
	parser.add_argument('--benchmark',			required=True)
	parser.add_argument('--dataset',			required=True,	help='Name of the dataset in the Data folder')
	parser.add_argument('--predictions_csv',	action='store_true',	help='Save predictions in csv format')
	parser.add_argument('--test_split',			type=int,	default=70)
	parser.add_argument('--results_folder',		default='results',	help='Folder to save results')
	return parser

def set_preprocess_args(parser):
	parser.add_argument('--input_counters',		default=['CPI'],	nargs='+',	help='List of model inputs. '\
												'The first counter in the list will be used as the ouput of '\
												'forecasting models')
	parser.add_argument('--pca',				help='Use PCA to reduce the dimensionality of the data. Use ' \
												'integer values for the number of components or a fraction for' \
												' the percentage of variance')
	parser.add_argument('--start_drop_count',	type=int,	default=0,	help='Number of samples to drop at the'\
												' beginning of the training set')
	parser.add_argument('--end_drop_count',		type=int, default=0, 	help='Number of samples to drop at the'\
												' end of the training set')
	parser.add_argument('--scaler',				default='minmax', choices=['minmax', 'standard', 'none', 'minmax01'])
	parser.add_argument('--filter',				default='none', choices=['median', 'none'])
	parser.add_argument('--filter_size',		type=int, default=3)
	return parser

def get_input_args(caller_file, input_args=None):
	parser = argparse.ArgumentParser()
	parser = set_experiment_args(parser)
	parser = set_preprocess_args(parser)
	
	if 'classify' in caller_file:
		parser.add_argument('--phase_count',			type=int, required=True)
		parser.add_argument('--classifier',				choices=['table', '2kmeans', 'pcakmeans', 'gmm'])
		parser.add_argument('--classifier_threshold',	type=float, default=1)
		parser.add_argument('--distance_metric',		choices=['euclidean', 'manhattan'])
		parser.add_argument('--W',						type=int,	default=100)
		parser.add_argument('--N1',						type=int,	default=10)
		parser.add_argument('--multicore_phases', 		choices=['local', 'global', 'local+shared'])

	if input_args:
		args = parser.parse_args(args=input_args)
	else:
		args = parser.parse_args()

	# Special argument types:
	# PCA
	if args.pca:
		if args.pca.isdigit():
			args.pca = int(args.pca)
		else:
			args.pca = float(args.pca)
	return args


