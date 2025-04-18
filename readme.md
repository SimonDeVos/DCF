# Decision-centric fairness: Evaluation and optimization for resource allocation problems</br><sub><sub>Simon De Vos, Jente Van Belle, Andres Algaba, Wouter Verbeke, Sam Verboven </sub></sub>  

This paper proposes a decision-centric fairness approach for binary classification models used in resource allocation. It argues that fairness constraints should be applied only within the decision-making region, where predicted scores translate into real-world actions, rather than across the entire score distribution, thereby achieving fairer outcomes with minimal loss in predictive utility.

A preprint is available on XXXX

The main contributions of this paper are as follows:
1. We introduce and formalize the concept of decision-centric fairness for resource allocation optimization.
2. We propose a decision-centric fairness approach to optimize classification models used in resource allocation.
3. We introduce a decision-centric predictive performance metric for classification models.
4. We empirically compare our proposed decision-centric fairness methodology to a global fairness approach on multiple (semi-synthetic) datasets, identifying scenarios where — from a decision-centric evaluation perspective — focusing on fairness only where it truly matters outperforms imposing fairness everywhere.

## Methodology  

<p align="left">
  <b>Optimization of decision-centric fairness:</b> The evolution of density distributions for different regularization strengths. 
  Solid lines represent full prediction densities, while dotted lines represent top-n selected predictions. 
    The decision threshold $\tau$ is indicated as a vertical line.
</p>

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/SimonDeVos/FairChurn/blob/master/examples/fig/density_0.5_0.7_0.0_True.gif" width="250">
      <br><b>λ = 0.0</b>
    </td>
    <td align="center">
      <img src="https://github.com/SimonDeVos/FairChurn/blob/master/examples/fig/density_0.5_0.7_0.3_True.gif" width="250">
      <br><b>λ = 0.3</b>
    </td>
    <td align="center">
      <img src="https://github.com/SimonDeVos/FairChurn/blob/master/examples/fig/density_0.5_0.7_0.6_True.gif" width="250">
      <br><b>λ = 0.6</b>
    </td>
  </tr>
</table>
</p>

<p align="center">
  The evolution of density distributions for different values of λ.  
</p>

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.


## Instructions
- Specify project configurations in [projectconfig.json](config/projectconfig.json)
- Specify experiment configurations in [experimentconfig.json](config/experimentconfig.json)
- In [main.ipynb](notebooks/main.ipynb):
  - Set the project directory to your custom folder. E.g., `DIR = r'C:\Users\...\...\...'`
  - Run [main.ipynb](notebooks/main.ipynb).
  - Results can be logged in the specified WandB project (more info on [WandB's website](https://docs.wandb.ai/quickstart/)). Configurations can be specified in `\DCF\config\projectconfig.json`
- Run [results.ipynb](notebooks/results.ipynb) to plot pareto curves similar to those in the paper. The results are imported from your configured WandB project

## Repository Structure
This repository is organized as follows:
```bash
|- config/
    |- experimentconfig.json    
    |- projectconfig.json    
|- data/
    |- adult/             
    |- TelecomKaggle/       
|- figures/
    |- 01_intro/             
    |- 03_methodology/       
    |- 04_experiment/             
    |- 05_results/
|- notebooks/ 
    |- main.ipynb
    |- results.ipynb            
|- src/
    |- data.py
    |- loss.py
    |- metrics.py
    |- model.py
    |- training.py
    |- utils.py
|- requirements.txt
```






