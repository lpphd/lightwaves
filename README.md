# Taking ROCKET on an Efficiency Mission: Multivariate Time Series Classification with LightWaveS

This repository contains the official implementation for the models described
in [Taking ROCKET on an Efficiency Mission: Multivariate Time Series Classification with LightWaveS](https://arxiv.org/abs/2204.01379)
.

If you find this work helpful in your research, consider citing our paper:

```
@article{pantiskas2022lightwaves,
  title={Taking ROCKET on an Efficiency Mission: Multivariate Time Series Classification with LightWaveS},
  author={Pantiskas, Leonardos and Verstoep, Kees and Hoogendoorn, Mark and Bal, Henri},
  journal={arXiv preprint arXiv:2204.01379},
  year={2022}
}
```

## Requirements

The code is written in Python 3.7.13 and has the following main dependencies for the training and evaluation scripts:

* numba==0.55.1
* sympy==1.10.1
* scipy==1.4.1
* sktime==0.10.1
* numpy==1.21.5
* scikit-learn==1.0.2
* pandas==1.3.5

The version of the [mpi4py](https://mpi4py.readthedocs.io/en/stable/index.html) library is 4.0.0.dev0 and has been
installed from its [repository](https://mpi4py.readthedocs.io/en/stable/install.html).

## Datasets

LightWaveS and (MINI)ROCKET are evaluated on the following classification datasets:

* [UEA multivariate dataset collection](https://www.timeseriesclassification.com/dataset.php)
* [MAFAULDA: Machinery Fault Database](http://www02.smt.ufrj.br/~offshore/mfs/page_01.html)
* [Turbofan Engine Degradation Simulation Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)

The InsectWingbeat dataset from the UEA collection has been excluded from the experiments since due to its large size it
prevented the ROCKET training process from completing successfully.

The MAFAULDA and TURBOFAN datasets have been suitably processed for the classification task. The code for this is in
the [PrepareDatasets.py](PrepareDatasets.py) file.

## Results

The accuracy metrics for ROCKET, MINIROCKET and the default LightWaveS variant (L1L2) on the above datasets can be
found [here](Metrics/Accuracy-LightWaveS-ROCKET-MINIROCKET.md).

The accuracy for the L1, L2 variants of LightWaveS, as well as the L1L2 variant with 1500 featuers, can be
found [here](Metrics/Accuracy-LightWaveS-L1-L2-L1L21500.md).

The inference speedup of LightWaveS variants over ROCKET can be found [here](Metrics/SpeedupOverROCKET.md) and over
MINIROCKET [here](Metrics/SpeedupOverMINIROCKET.md).

The number of channels required for the LightWaveS variants during inference, compared to the original number of
channels for each dataset, can be found [here](Metrics/ChannelReduction.md).
