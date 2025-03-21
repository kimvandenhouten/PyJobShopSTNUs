PyJobShop and Temporal Networks
====

# Available code
## PyJobShopIntegration
This folder provides a startpoint for this research project, as it shows some initial steps towards converting the original 
AAAI code to generalized code that can be integrated with PyJobShop. Note that the other folders in this repo origin from
the original AAAI github repository, for which the READ.me is located in the root folder.

### example_fjsp.py
This script uses a PyJobShop example for the FJSP, and converts it's solution into a STNU and runs the DC-checking algorithm.

### example_rcpsp_max.py
This script uses the uses an instance from RCPSP/max and our code, but now uses a PyJobShop model to solve the deterministic problem and to construct the STNU.

### rcpsp_example_pyjobshop.py
This script has and example from the PyJobShop package on the rcpsp.

### utils.py
This script contains a piece of code from Joost Berkhout which is close to what we have developed for get_resource_chains. I also copied our get_resource_chains there, because I expect that we will make adjustments to it.

# Proposed problems per student:
1. Flexible-job shop problem with no-wait constraints and maximal time-lags
2. Flexible-job shop problem with sequence dependent set-up times
3. Flexible-job shop problem with (hard) deadlines
4. Multi-mode RCPSP (optional with deadlines deadlines)
5. Multi-mode RCPSP/max (with generalized time-lags/no-wait constraints)


# Suggestions for benchmark instances 

Kolisch, R., & Sprecher, A. (1997). PSPLIB-a project scheduling problem library: OR software-ORSEP operations research software exchange program. European journal of operational research, 96(1), 205-216.

Reijnen, R., van Straaten, K., Bukhsh, Z., & Zhang, Y. (2023). Job shop scheduling benchmark: Environments and instances for learning and non-learning methods. arXiv preprint arXiv:2308.12794.

* It is expected that the instances need to be adjusted for the stochastic setting.


# Reading list 
Lan, L., & Berkhout, J. (2025). PyJobShop: Solving scheduling problems with constraint programming in Python. arXiv preprint arXiv:2502.13483.

Naderi, B., Ruiz, R., & Roshanaei, V. (2023). Mixed-integer programming vs. constraint programming for shop scheduling problems: new results and outlook. INFORMS Journal on Computing, 35(4), 817-843.

Houten, K. V. D., Planken, L., Freydell, E., Tax, D. M., & de Weerdt, M. (2024). Proactive and Reactive Constraint Programming for Stochastic Project Scheduling with Maximal Time-Lags. arXiv preprint arXiv:2409.09107.

Policella, N., Cesta, A., Oddi, A., & Smith, S. F. (2007). From precedence constraint posting to partial order schedules. Ai Communications, 20(3), 163-180.

Hunsberger, L., & Posenato, R. (2024). Foundations of Dispatchability for Simple Temporal Networks with Uncertainty. In ICAART (2) (pp. 253-263).

Morris, P. (2014, May). Dynamic controllability and Dispatchability relationships. In International Conference on Integration of Constraint Programming, Artificial Intelligence, and Operations Research (pp. 464-479). Cham: Springer International Publishing.




