# Degradation-path-prediction-of-lithium-ion-batteries-under-dynamic-operating-sequences

These are the codes for the paper: Degradation path prediction of lithium-ion batteries under dynamic operating sequences. Inwoo Kim, Jang Wook Choi

### Data preparation
Kim, Inwoo; Choi, Jang Wook (2024), “Degradation path prediction of lithium-ion batteries under dynamic operating sequences”, Mendeley Data, V1, doi: 10.17632/h2y7mj4kt7.1

### Code overview
The code is organized into two sections. 
  * The data processing section, which extracts model inputs from raw data
  * The data analysis section, which covers model training, evaluation, and plotting. 

The preprocessed data generated in the data processing section can be used in the data analysis section as model inputs. 

Since the raw data is limited to the 4th decimal place, the preprocessed data with precision up to the 6th decimal place has been additionally provided for reference.

* <mark>Data processing.ipynb</mark> : Collection of capacity and degradation index (DI) data from periodic reference performance tests (RPTs).
* <mark>Data analysis.ipynb</mark> : The pipeline for model training, evaluation, and plotting. 
* <mark>functions.py</mark> : Module for data analysis.

