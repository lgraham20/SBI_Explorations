import torch
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNRE, prepare_for_sbi, simulate_for_sbi

### Ipmorting Global Constants
### SNPE_vars contains global variables, and SNPE_func
### contains a few custom functions involved in the 
### Bayesian analaysis and plotting the results.
from SNRE_func import *
from SNRE_var import *

### If hard_mode is true, the mock observation will 
### be one that base SNPE w/ MAF struggled to
### identify as a Gaussian.
hard_mode = False

### The filepath determines where the resulting
### pairplots will be generated and named.
filepath = '3resnet_500Big.png'

### Defining our Prior
### This default prior is a uniform distribution, with the 
### same bounds on each parameter. 
prior = utils.BoxUniform(low=low*torch.ones(num_dim), high=high*torch.ones(num_dim)) #Creates our parameter space

### Creating a simulated data observation
### Normally, this will just draw a set of 
### random pairs of points from a 2D Gaussian
### distribution and takes their means. 
### If Hard Mode, it copies the bad 
### data set instead.
if not hard_mode:
    observation = theta_true + sigma_true*torch.randn((n_obs,num_dim)) #Notice, the same as our simulator below!
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
inference = SNRE(prior=prior)

### Creates a list to hold each posterior the neural net generates
### and defines are first proposal for the distribution as the prior.
posteriors = []
proposal = prior

### For an explanation on multiround inference, https://www.mackelab.org/sbi/tutorial/03_multiround_inference/
### In short, this loop trains our neural network across the entire proposed parameter space ("proposal"),
### then it is fed the observation which it uses to narrow the parameter space and focus in on the area
### closest to our observation.
for _ in range(n_runs):
    #Uses our simulator to generate a number (sim_count) of mock observations across our
    #proposed parameter space (prior).
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations= sim_count,num_workers=3) 
    
    #Adds the simulated data to the inference object (and trains it)
    density_estimator = inference.append_simulations(theta, x).train() 
    
    #Creates the posterior (posterior = p(theta | x))
    posterior = inference.build_posterior(density_estimator)
    #Adds that posterior to the list of posteriors
    posteriors.append(posterior)
    #Narrows our parameter space to ones the posterior thinks are likely given
    #our observation.
    proposal = posterior.set_default_x(observed_mean)
    #Then, loop repeats with the narrowed parameter space.
    print('Finished run ' + str(_ + 1) + '!')

fig = plot_chains(pdims = dimens_plot,posts = posteriors,obsv = observation,samps = sample_count)
plt.savefig(filepath)  
