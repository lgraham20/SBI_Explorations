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
hard_mode = False #Determines whether data is random or a specifically hard 2D data set
figID = '3'
#Defining our Prior
prior = utils.BoxUniform(low=low*torch.ones(num_dim), high=high*torch.ones(num_dim)) #Creates our parameter space

#Creating a simulated data observation
if not hard_mode:
    observation = theta_true + sigma_true*torch.randn((n_obs,num_dim)) #Notice, the same as our simulator below!
    observed_mean = torch.mean(observation,0)
    print('Here is the observed data: ' + str(observation) + '\n' + 'Here is the mean: ' + str(observed_mean))
else:
    observation = difficult_obs
    observed_mean = torch.mean(observation,0)
    print('Here is the observed mean: ' + str(observed_mean))

#Simulation: Suppose that observed data will be from Gaussian with StD=1
def simulator(theta):
    simobs = theta + sigma_true*torch.randn((n_obs,num_dim))
    sim_mean = torch.mean(simobs,0)
    return sim_mean

# Now it's time to begin the SBI.
simulator, prior = prepare_for_sbi(simulator, prior)
#dense_estimator = posterior_nn(model='maf', hidden_features=100, num_transforms=5)
inference = SNRE(prior=prior)

posteriors = []
#mcmcs = []
proposal = prior

#posterior1 = infer(simulator, prior, 'SNRE',num_workers=3,num_simulations=sim_count)
#posteriors.append(posterior1)

#potential1 = posterior_estimator_based_potential(posterior_estimator=posterior1,prior=prior,x_o=observed_mean)
#mcmc = MCMCPosterior(potential_fn=potential1,proposal=posterior1)
#mcmcs.append(mcmc1)


for _ in range(n_runs):
    #Runs simulations for our simulator, based on the prior + simulator
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations= sim_count,num_workers=3) 
    
    #Adds the simulated data to the inference object (and trains it)
    density_estimator = inference.append_simulations(theta, x).train() 
    
    #Creates the posterior (posterior = p(theta | x))
    posterior = inference.build_posterior(density_estimator)
    #Adds that posterior to the list of posteriors
    posteriors.append(posterior)
    #Trains around the specific observation
    proposal = posterior.set_default_x(observed_mean)
    print('Finished run ' + str(_) + '!')

fig = plot_chains(pdims = dimens_plot,posts = posteriors,obsv = observation,samps = 5000)
plt.savefig('SNRE_Randoms/'+ figID + 'SNREGaussian.png')  