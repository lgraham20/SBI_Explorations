# SBI Explorations
Experimenting with Simulation Based Inference (https://www.mackelab.org/sbi/) on theoretical and (in the future) real data.
Currently exploring how SBI compares to Bayesian analysis on Gaussian data.

## Organization
Currently I've written code to run the comparison between SBI and Bayesian analysis on random Gaussian data using two different neural inference algorithms, SNPE and SNRE. In my experience SNRE takes slightly longer to train, and significantly longer to generate sample data for pairplots but it is much better at learning. This is probably because it generates a density ratio instead of attempting to guess a direct posterior distribution (https://arxiv.org/pdf/2007.09114.pdf). 

The func.py and var.py files contain global variables or functions that are used by the SBI v. Gaussian scripts but not
the involved in the machine learning. Besides modifying the number of workers you want to generate simulations for SBI and the parameters for a custom density estimator, all the parameters you should need to change will be in the var.py files. Refer to the SNPE.py or SNRE.py (or https://www.mackelab.org/sbi/tutorial/04_density_estimators/) for more information on custom density estimators. 
