# stochasticRCPSPmax
This repository implements a proactive, a reactive, and a hybrid approach for stochastic RCPSP/max. It uses state-of-the-art techniques from Constraint Programming (CP) and Simple Temporal Networks with Uncertainty (STNUs).  


## Installation and practical issues
From a terminal session:
```shell
cd /path/to/stochasticRCPSPmax      # Go to the directory where you downloaded the repository
python3 -m venv venv                # Create a new virtual environment
. venv/bin/activate                 # Activate it
pip install -r requirements.txt     # Install dependencies
```
To be able to run all experiments in this repository, you also need to install the IBM CPLEX Optimization Studio (full edition, available via an academic licence).
The CPLEX optimizer must then be made available to the scripts by creating a symbolic link in the virtualenv; for example (adjust paths for your system):

```shell
ln -s /opt/ibm/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux/cpoptimizer /path/to/stochasticRCPSPmax/venv/bin/
```

For running scripts from this repository, make sure that:
* the working directory is `/path/to/stochasticRCPSPmax`
* the virtual environment is activated
* `/path/to/stochasticRCPSPmax` is in `PYTHONPATH

```shell
cd /path/to/stochasticRCPSPmax
. venv/bin/activate  
export PYTHONPATH=/path/to/stochasticRCPSPmax-main:$PYTHONPATH
```

## CSTNU tool
The STNU-based algorithms make use of the Java CSTNU Tool by Roberto Posenato[^1]. Our Python repository already includes a JAR file for running the CSTNU-Tool, so no further action is needed.

[^1]: **Posenato, R. (2022)**. *CSTNU Tool: A Java library for checking temporal networks.* SoftwareX, 17, 100905. [http://dx.doi.org/10.1016/j.softx.2021.100905]
  
## Reproduction of experiments and tables
To reproduce the results from our experiments, run:
```
python3 aaai25/experiments/run_experiments.py
```
To solve the deterministic instances with perfect information, run:
```
python3 aaai25/experiments/run_deterministic_instances.py
```
To reproduce the table with the comparison of the feasibility ratios, run:
```
python3 aaai25/generate_tables/compare_feasibility_ratios.py
```
To reproduce the tables that are provided in the supplementary material and that include the test results from the Wilcoxon test, Proportion test, and Magnitude test, run the following:
```
python3 aaai25/generate_tables/compare_two_methods_obj_to_latex.py
python3 aaai25/generate_tables/compare_two_methods_offline_time_to_latex.py
python3 aaai25/generate_tables/compare_two_methods_online_time_to_latex.py
```

Note that Table 1 of the main paper "Proactive and Reactive Constraint Programming for Stochastic Project Scheduling with Maximal Time-Lags" shows a subset of the results in the tables that are output by the scripts above.
