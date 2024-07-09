# stochasticRCPSPmax
This repository implements a proactive, reactive, and a hybrid approach for stochastic RCPSP/max. It uses state-of-the-art techniques from Constraint Programming (CP) and Temporal Planning, namely Simple Temporal Networks with Uncertainty (STNUs).  


## Installation and practical issues
Besides the installation of the requirements.txt, it is needed to install the IBM CPLEX Optimization Studio (full edition, available via an academic licence) in order to be able to run all experiments in this repository.
You may need to make the CPLEX optimizer available to the scripts by creating a symbolic link in the virtualenv, for example (adjust paths for your system):

```
ln -s /opt/ibm/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux
/cpoptimizer /path/to/stochasticRCPSPmax/venv/bin/
```

Note that all scripts in this repository require the working directory to 
be `/path/to/stochasticRCPSPmax`, and for this directory to be in 
`PYTHONPATH`.

## CSTNU tool
The STNU-based algorithms make use of the Java CSTNU tool* by Robert Posenato. This repository contains a JAR file that enables running the Java code from Python. Please refer to ... to find more information about the CSTNU tool.

* Posenato, R. (2022). CSTNU Tool: A Java library for checking temporal networks. SoftwareX, 17, 100905.
  
## Run experiments
To reproduce the results from our experiments run:
experiments/aaai25/experiments/run_experiments.py

To reproduce the tables that are provided in the supplementary material and that include the test results from the Wilcoxon test, Proportion test, and Magnitude test, run the following scripts:
- statistical_tests/compare_two_methods_obj_to_latex.py
- statistical_tests/compare_two_methods_offline_time_to_latex.py
- statistical_tests/compare_two_methods_online_time_to_latex.py
