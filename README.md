# phase-aware_pmu_forecasting
Scripts to train phase-aware models that forecast performance counters values

## Data

* We provide sample data in the [Data](Data) folder, but you can [collect your own PMU values](https://github.com/esalcort/collectPMUEmon)
* The [Data](Data) folder contains CSV files with already collected and formatted PMU data
    * In [set104](Data/set104/) we provide data collected from running [SPEC CPU 2017](https://www.spec.org/cpu2017/)  (with OpenMP)
    * In [set106](Data/set106/) we provide data collected from running [PARSEC 3](https://parsec.cs.princeton.edu/parsec3-doc.htm)


## Requirements and Installation
This code has been developed in Python 3. The [package-list.txt](package-list.txt) file contatins a list of Python libraries that are necessary to run the scripts. You can use your preferred Python package manager and virtual environment. Here's an example of how to set up an environment with [Anaconda](https://www.anaconda.com/):

```shell
conda create -n pmuForecasting python=3.10
conda activate pmuForecasting
conda install --file package-list.txt
```

## User Guide
Running the main scripts should follow the format:
```shell
python <script> --dataset <setname> --benchmark <bm> <args>
```
Where:
 * *script* is [classify.py](classify.py). (We will add support for other phase aware methods later, as indicated in [our plan](#work-in-progress).)
 * *setname* is a folder inside of [Data](Data)
 * *bm* is a CSV file containing PMU data and located inside *setname*.
 * *args* can be a list of required arguments specific to the *script* and/or optional arguments. Use *python \<script\> --help* for more information

If you want to generate an output CSV file with the predictions, use the *--predictions_csv* option and the *--results_folder* to indicate where to store the CSV file. Note that the default folder is *results*, and the script assumes that the folder already exists.


### Phase classification
To classify phases use the [classify.py](classify.py) script. This script has one additional required argument: *phase_count*, which indicates the number of phases to be classified.
The output of this script has a list per-CPU classification metrics.

We support three definitions of phases as defined in [1]: *global*, *local*, and *local+shared*. They can be set with the *--multicore_phases* input argument. We also support different classification models, including *2kmeans* and *fgmm*, as defined in [1]. The examples below classify the data in *Data/set104/644.nab_s0.csv*.
* *2kmeans* example that classifies data into 6 global phases, and uses a window size of 21 to define the sencond-level phases of *2kmeans*.
```bash
python classify.py --dataset set104 --benchmark nab --classifier 2kmeans --phase_count 6 --W 21 --multicore_phases global --input_counters CPI L2_RQSTS.MISS OFFCORE_REQUESTS.DEMAND_DATA_RD FP_ARITH_INST_RETIRED.SCALAR_DOUBLE BR_MISP_RETIRED.ALL_BRANCHES 
```
* *fgmm* example that classifies data into 4 phases per CPU, and uses a filter size of 21 before classifying with gaussian mixture models (GMM).
```bash
python classify.py --dataset set104 --benchmark nab --classifier gmm --phase_count 4 --filter_size 21 --multicore_phases local --input_counters CPI L2_RQSTS.MISS OFFCORE_REQUESTS.DEMAND_DATA_RD FP_ARITH_INST_RETIRED.SCALAR_DOUBLE BR_MISP_RETIRED.ALL_BRANCHES 
```

## Work in progress

The code is currently being cleaned and formatted to make it easier to use by people other than the authors. We have outlined a plan with release dates in stages below. Contact Susy at esalcort@utexas.edu if you need a copy of the code in advance.

| Feature | Estimated Release Date |
| --------| ---------------------- |
| ~~Multi-core phase classification~~ | ~~02/10/2023~~ |
| Multi-core basic forecasting (phase-unaware) | 02/21/2023 |
| Multi-core window-based phase prediction | 03/07/2023 |
| Multi-core phase-based forecasting | 03/21/2023 |
| Multi-core phase-aware forecasting | 04/04/2023 |
| Multi-core phase change prediction | 04/18/2023 |
| Multi-core phase duration prediction | 05/01/2023 |
| Single-core phase classification | 05/15/2023 |
| Single-core phase prediction | 05/29/2023 |
| Single-core phase-based forecasting | 06/12/2023 |


## References
[1] Erika S. Alcorta and Andreas Gerstlauer, "[Learning-based Phase-aware Multi-core CPU Workload Forecasting](http://doi.org/10.1145/3564929)", ACM Transactions on Design Automation of Electronic Systems, 28(2):23:1–23:27, March 2023