# IncrementalPruning

[![Build Status](https://travis-ci.org/JuliaPOMDP/IncrementalPruning.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/IncrementalPruning.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/IncrementalPruning.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaPOMDP/IncrementalPruning.jl?branch=master)

This Julia package implements the incremental pruning solver for partially observable Markov decision processes.

## Installation

```julia
Pkg.clone("https://github.com/JuliaPOMDP/IncrementalPruning.jl")
```

## Usage

```julia
using IncrementalPruning
using POMDPModels
pomdp = TigerPOMDP() # initialize POMDP

solver = PruneSolver() # set the solver

policy = solve(solver, pomdp) # solve the POMDP  
```
The result of `solve` is a `Policy` that contains the alpha vectors of the solution.

IncrementalPruning.jl solves problems implemented using the [POMDPs.jl interface](https://github.com/JuliaPOMDP/POMDPs.jl). See the [documentation for POMDPs.jl](http://juliapomdp.github.io/POMDPs.jl/latest/) for more information.

## Algorithm Details

This solver implements the incremental pruning algorithm as described in Zhang and Liu (1996) and Cassandra et al. (1997). This solution method is exact (ϵ-optimal) but is much slower than modern approximate solution techniques. As such, it is only computationally feasible for small problems.

## References

Cassandra, A., Littman, M., & Zhang, N. (1997). Incremental pruning: A simple, fast, exact method for partially observable Markov decision processes. Proceedings of the Thirteenth Annual Conference on Uncertainty in Artificial Intelligence (UAI-97), 54–61.

Zhang N. L., Liu W. (1996). Planning in stochastic domains: Problem characteristics and approximation. Technical Report HKUST-CS96-31, Hong Kong University of Science and Technology.
