# Direct ITR optimisation for SSVEP-based BCI

## Introduction

This repository contains code for optimising information transfer rate (ITR) in steady-state visual evoked potential (SSVEP) based brain computer interfaces (BCIs). This approach should work in other types of BCIs as well. The novelty of the proposed classification method is that it is based on direct ITR maximisation. ITR is a standard measure of performance for BCIs. It combines the accuracy and the speed of the classifier into a single number which shows how much information is transferred by the BCI in one unit of time. Therefore, maximising ITR maximises the amount of information that the user can transfer to an external device (computer, robot, etc) in a fixed time interval.

The algorithm is introduced in the [article](https://iopscience.iop.org/article/10.1088/1741-2552/aae8c7): Anti Ingel, Ilya Kuzovkin, and Raul Vicente. "Direct information transfer rate optimisation for SSVEP-based BCI". Journal of neural engineering 16.1 (2018). Please cite this article when using the code.

A few improvements to the original method were suggested in the [article](https://doi.org/10.3389/fnhum.2021.675091): Anti Ingel and Raul Vicente. "Information Bottleneck as Optimisation Method for SSVEP-Based BCI".  Frontiers in Human Neuroscience 15 (2021). These improvements have been implemented in this repository.

## Requirements

The code is written in Python and has been tested with Python 3.7. Running the code requires packages `numpy`, `scipy`, `sklearn`, `pandas`, `matplotlib`, `sympy`.

## Getting Started

Before trying to run the code, please download the required dataset from [here](https://drive.google.com/drive/folders/12Wu6377sfgYZ2qpOw_WUtgVYO97maDlu?usp=sharing). The original dataset can be found [here](http://www.bakardjian.com/work/ssvep_data_Bakardjian.html), but this is not necessary to run the code.

Download `data` folder to the same folder that contains `src`. The `data` folder contains the features extracted from the original data. The extracted features can be calculated from the original data by running the files `1_original_data_to_csv.py`, `2_generate_eeg_data.py`, `3_generate_feature_data.py`. If the features have been obtained (either by downloading directly, or calculating from the original data), the optimisation procedure can be executed by running the file `4_optimise_itr.py`.

## Contact

For additional information, feel free to contact Anti Ingel (antiingel@gmail.com).
