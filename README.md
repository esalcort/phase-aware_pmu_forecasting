# phase-aware_pmu_forecasting
Scripts to train phase-aware models that forecast performance counters values

## Data

* Get started by [collecting PMU values](https://github.com/esalcort/collectPMUEmon)
* The [Data](Data) folder contains CSV files with already collected and formatted PMU data
    * In [set104](Data/set104/) we provide data collected from running [SPEC CPU 2017](https://www.spec.org/cpu2017/)  (with OpenMP)
    * In [set106](Data/set106/) we provide data collected from running [PARSEC 3](https://parsec.cs.princeton.edu/parsec3-doc.htm)


## Work in progress

The code is currently being cleaned and formatted to make it easier to use by people other than the authors. We have outlined a plan with release dates in stages below. Contact Susy at esalcort@utexas.edu if you need a copy of the code in advance.

| Feature | Estimated Release Date |
| --------| ---------------------- |
| Multi-core phase classification | 02/10/2023 |
| Multi-core basic forecasting (phase-unaware) | 02/21/2023 |
| Multi-core window-based phase prediction | 03/07/2023 |
| Multi-core phase-based forecasting | 03/21/2023 |
| Multi-core phase-aware forecasting | 04/04/2023 |
| Multi-core phase change prediction | 04/18/2023 |
| Multi-core phase duration prediction | 05/01/2023 |
| Single-core phase classification | 05/15/2023 |
| Single-core phase prediction | 05/29/2023 |
| Single-core phase-based forecasting | 06/12/2023 |


