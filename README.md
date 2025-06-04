# Ceramic Network 

## Description
Kroon, E.J. in his PhD thesis tried to reconstruct the interaction between two cultures: Funnel Beaker West (4000-2800 BCE)  and Corded Ware (3000-2350 BCE). In his thesis he uses a probabilistic approach and to determine the similarity he uses the wasserstein distance between the empirical pdfs of the edges, because from the dataset we can determine the probability to do a step after anoter in a certain culture: 
$p(x_{t+1}| x_{t})$.  
Here we try new approaches. 

## Installation

To set up the environment required to run this project, use [Conda](https://docs.conda.io/). The dependencies are specified in the `environment_full.yml` file.

To run the code clone the repository and create a conda enviroment with the dependencies. 

```bash 
conda env create --name env_name -f environment_full.yml
```

In the Presentation Notebook there is a self contained narrative of the project. 

In functions.py there are all the necessary developed tools to make the analysis
