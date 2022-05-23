from SNPE_vars import *
import numpy as np
from chainconsumer import ChainConsumer
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

#######################################################################
#Here I'm defining the pair plot maker, creates a chain               #
#of classic Bayesian analysis vs. SBI analysis given an SBI posterior,#
#an observation, and a number of samples to draw.                     #
#######################################################################
def pairplot_comp(post,obs,ns):
    ### Resets or Chain Consumer variable (might not be necessary)
    c =[] 
    ### Takes the mean of our mock observation
    obs_mu = torch.mean(obs,0) 

    ### Generates the SBI sample data from the posterior distribution.
    ### The specific chain consumer function used has problems with 
    ### torch, so it has to be converted into a numpy array.
    sbi_data = np.array(post.sample((ns,), x=obs_mu))

    ### Generating the Bayesian pairplot samples
    ### Covariance matrix based on data
    cov = torch.cov(obs.T)/np.sqrt(n_obs)
    ### m defins a Gaussian random generator from Torch
    ### based on our observed mean + covariance
    m = MultivariateNormal(obs_mu,cov)
    bsamps = torch.ones(50000,2)
    ### Pulling ns random points from the Bayesian guess, 
    ### converted to Numpy for same reason above.
    for el in range(50000):
        bsamps[el] = m.sample()
    bsamps = np.array(bsamps)
    c = ChainConsumer()
    #Adding our samples to the chain, returning the chain
    c.add_chain(sbi_data,parameters=["$x_1$", "$x_2$"],name='SBI')
    c.add_chain(bsamps,parameters=["$x_1$", "$x_2$"],name='Bayesian')
    return c

#####################################################################
#The following function makes chain consumer plots                  #
#####################################################################

### Dimensions of plot, list of posteriors, Torch observation (must be shape [N,2]), 
### sample size for sampling from posterior for pairplots.
### Depending on the dimensions of plot (whether axs is 1D, 2D, or scalar), 
### a slightly different plotting method is used.
def plot_chains(pdims,posts,obsv,samps):
    i = 0
    #Clearing cc, just in case
    cc = [] 
    if pdims[0] > 1:
        j = 0
        t = 1
        fig, axs = plt.subplots(pdims[0],pdims[1],figsize=(15,15),sharex='col',sharey='row')
        for el in posts:
            #Creating cc and plotting it onon a subplot axs
            #For info on ChainConsumer, see: https://samreay.github.io/ChainConsumer/chain_api.html
            #Using the plot_contour function because it integrates with Matplotlib better.
            cc = pairplot_comp(el,obsv,samps)
            cc.plotter.plot_contour(axs[i,j],'$x_1$','$x_2',chains=['SBI','Bayesian']);
            #Formatting plots
            axs[i,j].set_xlabel('X1')
            axs[i,j].set_ylabel('X2')
            axs[i,j].set_aspect('equal','box')
            axs[i,j].plot(theta_true*np.ones(np.shape(np.linspace(-5,5))),np.linspace(-5,5),'black')
            axs[i,j].plot(np.linspace(-5,5),theta_true*np.ones(np.shape(np.linspace(-5,5))),'black')
            axs[i,j].set_title('Iteration ' + str(t) )

            j+=1
            t+=1
            #Move to the next row when needed
            if j%pdims[1] == 0:
                j = 0
                i+=1
        return fig
    if pdims[0] == 1 and pdims[1] > 1:
        fig, axs = plt.subplots(pdims[0],pdims[1],figsize=(15,15),sharex='col', sharey='row')
        for el in posts:
            #Creating and plotting CC
            cc = pairplot_comp(el,obsv,samps)
            cc.plotter.plot_contour(axs[i],'$x_1$','$x_2$',chains=['SBI','Bayesian']);
            #Formatting
            axs[i].set_xlabel('X1')
            axs[i].set_ylabel('X2')
            axs[i].plot(theta_true*np.ones(np.shape(np.linspace(-5,5))),np.linspace(-5,5),'black')
            axs[i].plot(np.linspace(-5,5),theta_true*np.ones(np.shape(np.linspace(-5,5))),'black')
            axs[i].set_title('Iteration ' + str(i+1))
            print('Plotted plot ' + str(i) + '!')
            i+=1
        return fig
    if pdims[0] ==1 & pdims[1] ==1:
        fig, axs = plt.subplots(pdims[0],pdims[1],figsize=(15,15),sharex='col', sharey='row')
        for el in posts:
            #Creating and plotting CC
            cc = pairplot_comp(el,obsv,samps)
            cc.plotter.plot_contour(axs,'$x_1$','$x_2$',chains=['SBI','Bayesian']);
            #Formatting
            axs.set_xlabel('X1')
            axs.set_ylabel('X2')
            axs.plot(theta_true*np.ones(np.shape(np.linspace(-5,5))),np.linspace(-5,5),'black')
            axs.plot(np.linspace(-5,5),theta_true*np.ones(np.shape(np.linspace(-5,5))),'black')
            axs.set_title('Iteration ' + str(i+1))
            i+=1
    #The extent of my error handling :^ )
    else:
        print('Wrong dimensions!')
        return
