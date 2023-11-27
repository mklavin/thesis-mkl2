# thesis-mkl2
The goal of this project is to determine the concentrations of reduced (GSH) and
oxidized (GSSG) glutathione based on their Raman signals. 

# py files:
## data_collection
functions for quickly organizing raw data into a usable format.

## modeling
ML model training and testing.

## preprocessing
functions for preprocessing data, like normalizing, standardizing etc. 

## stratedgy_seach
uses some functions from preprocessing for a search algorithm which finds
the best permutation of preprocessing methods and the best baseline removal 
function out of ~30 pybaslines functions. Baselines are evaluated based on the 
covariance between peak points and labels which would ideally be 1. 

## visualize
plotting functions. 

# folders:
## data 
contains Raman data- daniels raw Raman data, PCA processed data, Mimi's 
raw Raman data, and preprocessed data.

## models
will contain finished models

## plots
contains images of plots used for presentations or written work.
