# name of files for landuse before and after the sample period. 
# All other files in folder are considered features
before_name = "Landuse2008"
after_name = "Landuse2018"

# Choose the model to run. LR for logistic regression, GWR for Geographically-weighted regression
model_name = "GWR" 


# Folder with data files
data_folder = "data"

# Folder to put results in
results_folder = "results"

# NOTE: don't choose data_folder and results_folder the same, program will break