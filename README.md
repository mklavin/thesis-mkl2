# thesis-mkl2
The goal of this project is to determine the concentrations of reduced (GSH) and
oxidized (GSSG) glutathione based on their Raman signals. Glutathione is an antioxidant which neutralizes free radicals
by cycling between its reduced and oxidized forms. 
Increased amounts of free radicals in the body is known as oxidative stress and is correlated with 
higher risk for a number of diseases. Determining the ratio of reduced to oxidized glutathione can therefore describe
an organisms state of oxidative stress for applications in disease prevention. 

key ideas:
* The integration of a peak in Raman spectra is directly proportionate to the concentration of its corresponding
  bonds. 
* The GSSG peak appears in the "580" region and the GSH peak appears in the "610" region. 580 & 610 denote the two areas Raman
  spectra are taken.
* These regions are separated into two datasets and models are trained on both separately. 


## py files:
### data_collection
functions for quickly organizing raw data into Pandas DataFrames.

### modeling
ML model training and testing.

### preprocessing
functions for preprocessing data, like normalizing, standardizing etc. 

### stratedgy_seach
uses some functions from preprocessing for a search algorithm which finds
the best permutation of preprocessing methods and the best baseline removal 
function out of ~30 pybaslines functions. Baselines are evaluated based on the 
covariance between peak points and labels which would ideally be 1. 

### visualize
plotting functions. 

## folders:
### data 
contains Raman data- daniels raw Raman data, PCA processed data, Mimi's 
raw Raman data, and preprocessed data. 

### models
will contain finished models

### plots
contains images of plots used for presentations or written work.
