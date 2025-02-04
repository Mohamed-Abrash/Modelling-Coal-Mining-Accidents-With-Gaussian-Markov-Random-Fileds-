# Modelling Coal Mining Accidents With Gaussian Markov Random Fileds

<img src="https://github.com/user-attachments/assets/b459526c-d535-4bce-b326-77ac9cb1e14c" alt="GRF_SAR" width="700" />

 ---

This project analyzes the number of coal mining accidents in Britain between 1750 and 1980, using a **Bayesian hierarchical model** with a **Gaussian Markov Random Field** (GMRF).  The research is motivated by the need to analyze and understand the patterns and trends in coal mining accidents over time, which can help in developing better safety measures and prevention strategies.  

The aim of the project is to model the number of accidents per year using a **Poisson distribution** with a **latent GMRF**.  The GMRF incorporates spatial and temporal correlations, allowing for a more nuanced understanding of the factors influencing accident rates.  The project employs **Laplace approximation** to estimate model parameters and reconstruct the latent field.  

The analysis includes comparing different GMRF structures (CAR and SAR models) and evaluating their performance in capturing the observed data.  The project also investigates the sensitivity of the model to parameter perturbations and the relationship between the variance of predictions and the observed number of accidents. 

This repository includes:
- Full written report
- MATLAB implementation
