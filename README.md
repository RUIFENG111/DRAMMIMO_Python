# DRAMMIMO
Delayed Rejection Adaptive Metropolis Multi Input Multi Output

This package utilizes a modified version of the Delayed Rejection Adaptive Metropolis (DRAM) algorithm to realize the Maximum Entropy (ME) method numerically. 

The original DRAM algorithm is based on the MATLAB toolbox from Dr. Marko J. Laine (https://mjlaine.github.io/mcmcstat/) and the book [1] from Dr. Ralph C. Smith. It uses the Markov Chain Monte Carlo (MCMC) method.

The Maximum Entropy method can be used for fusing data from hetergeneous sources and quantifying uncertainty of model parameters that are shared among models [2]. When only one set of data is available, the Maximum Entropy method automatically becomes the Bayesian method, and the algorithm is equivalent to the original DRAM algorithm.

The package provides code for both MATLAB and Python environment at this point. C/C++ version will come in the future. Inside each version, there is currently only one simple linear model to demonstrate how to use the package. More complicated examples will be added in the future.

[1] Ralph C Smith. Uncertainty quantification: theory, implementation, and applications, volume 12. Siam, 2013.

[2] Wei Gao, William S Oates, and Ralph C Smith. A maximum entropy approach for uncertainty quantification and analysis of multifunctional materials. In ASME 2017 Conference on Smart Materials, Adaptive Structures and Intelligent Systems, pages V001T08A013-V001T08A013. American Society of Mechanical Engineers, 2017.
