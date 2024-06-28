# stochasticRCPSPmax
This repository implements a proactive, reactive, and a hybrid approach for stochastic RCPSP/max. It uses state-of-the-art techniques from Constraint Programming (CP) and Temporal Planning, namely Simple Temporal Networks with Uncertainty.  


## Installation and practical issues
Besides the installation of the requirements.txt, it is needed to install the IBM CPLEX Optimization Studio (full edition, available via an academic licence) in order to be able to run all experiments in this repository.
You may need to make the CPLEX optimizer available to the scripts by creating a symbolic link in the virtualenv, for example (adjust paths for your system):

```
ln -s /opt/ibm/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux/cpoptimizer /path/to/Learning-From-Scenarios-for-Repairable-Stochastic-Scheduling/venv/bin/
```

Note that all scripts in this repository require the working directory to be `/path/to/Learning-From-Scenarios-for-Repairable-Stochastic-Scheduling`, and for this directory to be in `PYTHONPATH`.