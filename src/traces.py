# Utility functions to read and write traces
import pandas as pd
import os

DEFAULT_FOLDER = 'Data'
class benchmark_name:
	def __init__(self, name, number, refs):
		self.name = name
		self.number = number
		self.refs = refs

BENCHMARKS = {  'perlbench' : benchmark_name('perlbench_s', '600', ['0', '1', '2']),
				'mcf'       : benchmark_name('mcf_s',       '605', ['0']),
				'cactuBSSN' : benchmark_name('cactuBSSN_s', '607', ['0']),
				'lbm'       : benchmark_name('lbm_s',       '619', ['0']),
				'wrf'		: benchmark_name('wrf_s',		'621', ['0']),
				'x264'      : benchmark_name('x264_s',      '625', ['0', '1', '2']),
				'pop2'      : benchmark_name('pop2_s',      '628', ['0']),
				'deepsjeng' : benchmark_name('deepsjeng_s', '631', ['0']),
				'imagick'	: benchmark_name('imagick_s',	'638', ['0']),
				'leela'     : benchmark_name('leela_s',     '641', ['0']),
				'nab'       : benchmark_name('nab_s',       '644', ['0']),
				'exchange2' : benchmark_name('exchange2_s', '648', ['0']),
				'fotonik3d' : benchmark_name('fotonik3d_s', '649', ['0']),
				'roms'      : benchmark_name('roms_s',      '654', ['0']),
				'specrand_f': benchmark_name('specrand_fs', '996', ['0']),
				'specrand_i': benchmark_name('specrand_is', '998', ['0']),
				'xz'		: benchmark_name('xz_s',		'657', ['0', '1']),
				'synthetic' : benchmark_name('synthetic_s', '111', ['0']) # Synthetic benchmark to test transforms
			}

PARSEC_BENCHMARKS = [
						'blackscholes', 'canneal', 'ferret', 'fluidanimate', 'bodytrack',
						'freqmine', 'raytrace', 'streamcluster', 'vips', 'swaptions', 'facesim',
					]

# For normalizing data, most of the times only one counter is needed. But, there are
# a few exceptions. Show all known counters
fixed_counters = ['INST_RETIRED.ANY', 'CPU_CLK_UNHALTED.THREAD', 'CPU_CLK_UNHALTED.REF_TSC']
single_norm_counters = {
	'CPU_CLK_UNHALTED.THREAD' 				:	'CPI',
	'MEM_LOAD_UOPS_MISC_RETIRED.LLC_MISS'	:	'LLC_MPI',
	'BR_MISP_EXEC.ALL_BRANCHES'				:	'BR_MPI',
	'BR_INST_RETIRED.ALL_BRANCHES'			:	'BR_PI',	# 8 var counters
	'FP_COMP_OPS_EXE.X87'					:	'FP_X87_PI',
	'FP_COMP_OPS_EXE.SSE_PACKED_DOUBLE'		:	'SSE_PCKD_PI',
	'FP_COMP_OPS_EXE.SSE_SCALAR_SINGLE'		:	'SSE_SCLS_PI',
	'FP_COMP_OPS_EXE.SSE_PACKED_SINGLE'		:	'SSE_PCKS_PI',
	'FP_COMP_OPS_EXE.SSE_SCALAR_DOUBLE'		:	'SSE_SCLD_PI',
	'SIMD_FP_256.PACKED_SINGLE'				:	'SIMD_FP_256S_PI',
	'SIMD_FP_256.PACKED_DOUBLE'				:	'SIMD_FP_256D_PI',
	'ARITH.FPU_DIV'							:	'FPU_DIV_PI',
	'DTLB_LOAD_MISSES.MISS_CAUSES_A_WALK'	:	'DTLB_LD_MPI',
	'DTLB_STORE_MISSES.MISS_CAUSES_A_WALK'	:	'DTLB_ST_MPI',
	'ITLB_MISSES.MISS_CAUSES_A_WALK'		:	'ITLB_MPI',
	'MEM_UOPS_RETIRED.STLB_MISS_LOADS'		:	'STLB_LD_MPI',
	'ICACHE.MISSES'							:	'ICACHE_MPI',
	'RESOURCE_STALLS.ANY'					:	'RESOURCE_STALLS_PI',
	'OFFCORE_REQUESTS.ALL_DATA_RD'			:	'OFFCORE_DATA_RD_PI',
	'UOPS_RETIRED.ALL'						:	'UOPS_RETIRED_PI',
	'L2_RQSTS.ALL_DEMAND_DATA_RD'			:	'L2_DEMAND_RD_PI',
	'L2_RQSTS.DEMAND_DATA_RD_HIT'			:	'L2_DEMAND_RD_HPI',
	'CYCLE_ACTIVITY.CYCLES_L2_PENDING'		:	'MEM_CPI',
	'L1D_PEND_MISS.L2_STALL'				:	'L2_STALL_PI',
	'FP_ARITH_INST_RETIRED.SCALAR_DOUBLE'	:	'FP_SCLD_PI',		# 8 var counters
	'FP_ARITH_INST_RETIRED.SCALAR_SINGLE'	:	'FP_SCLS_PI',		# 8 var counters
	'UOPS_ISSUED.STALL_CYCLES'				:	'UOPS_STALLS_PI',		# 8 var counters
	'CYCLE_ACTIVITY.CYCLES_MEM_ANY'			:	'MEM_CPI',			# 8 var counters
	'CYCLE_ACTIVITY.STALLS_MEM_ANY'			:	'MEM_STALLS_PI',	# 8 var counters
	'BR_MISP_RETIRED.ALL_BRANCHES'			:	'BR_MPI',		# 8 var counters
	'OFFCORE_REQUESTS.L3_MISS_DEMAND_DATA_RD' : 'L3_MPI',
	'UOPS_EXECUTED.CORE'					:	'UOPS_PI',		# 8 var counters
	'FP_ARITH_INST_RETIRED.SCALAR'			:	'FP_SCL_PI',
	'FREERUN_PKG_ENERGY_STATUS'				:	'PKG_ENERGY_PI',
	'L2_RQSTS.DEMAND_DATA_RD_MISS'			:	'L2_RD_MPI',
	'LONGEST_LAT_CACHE.MISS'				:	'LLC_MPI',
	'L2_RQSTS.MISS'							:	'L2_MPI',
	'OFFCORE_REQUESTS.DEMAND_DATA_RD'		:	'OFFCORE_DATA_RD_PI',
}
single_rate_counters = {
	'L2_RQSTS.DEMAND_DATA_RD_HIT'	:	['L2_RQSTS.ALL_DEMAND_DATA_RD', 'L2_HIT_RATE'],
	'BR_MISP_EXEC.ALL_BRANCHES'		:	['BR_INST_RETIRED.ALL_BRANCHES', 'BR_MISS_RATE'],
	'BR_MISP_RETIRED.ALL_BRANCHES'  :	['BR_INST_RETIRED.ALL_BRANCHES', 'BR_MISS_RATE'],
	'INST_RETIRED.ANY'				:	['CPU_CLK_UNHALTED.THREAD', 'IPC']
}

def get_raw_data(benchmark, dataset, ref=0, folder=DEFAULT_FOLDER):
	# This method reads a CSV file (and it's not compatible with emon format). It
	# returns a pandas Data frame
	if benchmark in BENCHMARKS:
		bm =  BENCHMARKS[benchmark]
		filename = bm.number + '.' + bm.name + str(ref) + '.csv'
		if str(ref) not in bm.refs:
			print('Introduce a valid reference for %s. Available refs:'%(benchmark))
			print( bm.refs )
			return None
	elif benchmark in PARSEC_BENCHMARKS:
		filename = benchmark + '.csv'
	else:
		print('Data for benchmark %s not available'%(benchmark))
		print('Available benchmarks:')
		print(BENCHMARKS.keys())
		print(PARSEC_BENCHMARKS)
		return None
	set_folder = os.path.join(folder, dataset)
	path = os.path.join(set_folder, filename)
	multicore = False
	with open(path) as f:
		firstline = f.readline().rstrip()
		cpu = firstline.count('CPU')
		coma = firstline.count(',')
		if abs(cpu - coma) < 2:
			multicore = True
	if multicore:
		data = pd.read_csv(path, header=[0,1])
	else:
		data = pd.read_csv(path, header=0)
	return data

def get_processed_data(data):
	if data.columns.nlevels > 1:
		percore = []
		corenames = data.columns.get_level_values(0).unique()
		for core in corenames:
			df = get_processed_data(data[core])
			percore.append(df)
		return pd.concat(percore, axis=1, keys=corenames)
	proc_data = pd.DataFrame()
	missing = []
	for value in data.columns.values:
		if value in single_norm_counters:
			name = single_norm_counters[value]
			proc_data[name] = data[value] / data['INST_RETIRED.ANY']
		elif value not in fixed_counters:
			missing.append(value)
	if 'L2_RQSTS.ALL_DEMAND_DATA_RD' in data.columns.values:
		if 'L2_RQSTS.DEMAND_DATA_RD_HIT' in data.columns.values:
			proc_data['L2_MPI'] = (data['L2_RQSTS.ALL_DEMAND_DATA_RD'] - data['L2_RQSTS.DEMAND_DATA_RD_HIT'])/data['INST_RETIRED.ANY']
	for counter in single_rate_counters.keys():
		if counter in data.columns.values:
			pair = single_rate_counters[counter][0]
			if pair in data.columns.values:
				name = single_rate_counters[counter][1]
				proc_data[name] = data[counter] / data[pair]
	if len(missing) > 0:
		print('WARNING: some counters are unknown and not normalized')
		print(missing)
	return proc_data
