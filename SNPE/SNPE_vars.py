import torch

#Defining true variables
theta_true = 2 #The true value of the mean
sigma_true = 0.5 #The true value of the standard deviation

#Defining SBI variables
n_obs = 10 #The number of observations
num_dim = 2 #Number of variables
low = -5 #High/low bound on variables
high = 5
n_runs = 6 #Number of multi-round inference roudns
dimens_plot = [1,6] #Must multiply to equal n_runs
sim_count = 200
if dimens_plot[0]*dimens_plot[1] != n_runs:
    raise ValueError("Dimensions of plot don't multiply to n_runs.")
####################################################################
#Here is the not as great observation that was 
# messing up SBI with SNPE & MAF earlier.
difficult_obs = torch.tensor([[2.1920, 2.3860],
        [1.8902, 2.4567],
        [1.8280, 1.8631],
        [1.9814, 1.9449],
        [1.4256, 1.5735],
        [1.7206, 2.1204],
        [1.9613, 3.1238],
        [2.5764, 2.1602],
        [1.2950, 1.8468],
        [2.6323, 2.3226]])
