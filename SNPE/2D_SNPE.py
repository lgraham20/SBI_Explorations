import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


### Ipmorting Global Constants
### SNPE_vars contains global variables, and SNPE_func
### contains a few custom functions involved in the 
### Bayesian analaysis and plotting the results.
from SNPE_func import *
from SNPE_vars import *

### If hard_mode is true, the mock observation will 
### be one that base SNPE w/ MAF struggled to
### identify as a Gaussian.
hard_mode = False 

### The filepath determines where the resulting
### pairplots will be generated and named.
filepath = 'Outputs/SNPE_Trials/MDN/1000/1_1000_Big.png'

### Defining our Prior
### This default prior is a uniform distribution, with the 
### same bounds on each parameter. 
prior = utils.BoxUniform(low=low*torch.ones(num_dim), high=high*torch.ones(num_dim))

### Creating a simulated data observation
### Normally, this will just draw a set of 
### random pairs of points from a 2D Gaussian
### distribution and takes their means. 
### If Hard Mode, it copies the bad 
### data set instead.
if not hard_mode:
    observation = theta_true + sigma_true*torch.randn((n_obs,num_dim))
    observed_mean = torch.mean(observation,0)
    print('Here is the observed data: ' + str(observation) + '\n' + 'Here is the mean: ' + str(observed_mean))
else:
    observation = difficult_obs
    observed_mean = torch.mean(observation,0)
    print('Here is the observed mean: ' + str(observed_mean))

### This is the simulator that our SBI will use to generate simulated data 
### and train the neural net. Much like our mock observation, it takes 
### in a tensor of means, draws 10 random points from a 2D Gaussian 
### centered on the means, and then averages them together for 
### a summary statistic. 
def simulator(theta):
    simobs = theta + sigma_true*torch.randn((n_obs,num_dim))
    sim_mean = torch.mean(simobs,0)
    return sim_mean


###########################################
###########################################
## Now it's time to begin the SBI. ########
###########################################
###########################################

### Checks that our prior and simulator will work with SBI.
simulator, prior = prepare_for_sbi(simulator, prior)

##########################################################################################
###################### Density Estimator Info ############################################
### If you'd like to modify the meta-parameters, change the dense_estimator            ###
### variable and plug it in for density_estimator in the inference definition.         ###
### Alternatively, the density_estimator accepts a few prebuilt configurations         ###
### identified with strings.                                                           ###
### Use Neural Spline Flow with 'nsf' and Masked Autoregressive Flow with 'maf'.       ###
### 'mdn' and 'made' are also available. They have not been tested.                    ###
### If you don't specify a density estimator, the default 'maf' will be chosen.        ###
###                                                                                    ###
### The posterior_nn() function may also be imported and used to design                ###
### a custom probability density estimator.                                            ###
### from sbi.utils.get_nn_models import posterior_nn                                   ###
### dense_estimator = posterior_nn(model='nsf', hidden_features=100, num_transforms=10)###
##########################################################################################
##########################################################################################

### Defines our inference object, which represents the neural network.
inference = SNPE(prior=prior,density_estimator='mdn')

### Creates a list to hold each posterior the neural net generates
### and defines are first proposal for the distribution as the prior.
posteriors = []
proposal = prior

### For an explanation on multiround inference, https://www.mackelab.org/sbi/tutorial/03_multiround_inference/
### In short, this loop trains our neural network across the entire proposed parameter space ("proposal"),
### then it is fed the observation which it uses to narrow the parameter space and focus in on the area
### closest to our observation.
for _ in range(n_runs):
    print('\nBeginning run ' + str(_) + '\n')
    #Uses our simulator to generate a number (sim_count) of mock observations across our
    #proposed parameter space.
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations= sim_count,num_workers=2) 
    
    #Adds the simulated data to the inference object and trains the neural net on it.
    density_estimator = inference.append_simulations(theta, x, proposal=proposal).train() 
    
    #Creates the posterior (posterior = p(theta | x), probability of a parameter given observation)
    posterior = inference.build_posterior(density_estimator)
    #Adds that posterior to the list of posteriors
    posteriors.append(posterior)
    #Narrows our parameter space to ones the posterior thinks are likely given
    #our observation.
    proposal = posterior.set_default_x(observed_mean)
    #Then, loop repeats with the narrowed parameter space.

### Plotting and saving our results.
### In the resulting plot, the green static pairplot is the Bayesian approximation,
### while the blue evolving plot is the SBI approximation. Usually it converges somewhat
### towards the Bayesian. 
fig = plot_chains(pdims = dimens_plot,posts = posteriors,obsv = observation,samps = sample_count)
plt.savefig(filepath)  
