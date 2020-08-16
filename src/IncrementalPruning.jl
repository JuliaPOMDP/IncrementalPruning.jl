module IncrementalPruning

using LinearAlgebra
using POMDPs, POMDPModelTools, POMDPPolicies
using POMDPLinter: @POMDP_require
using JuMP, GLPK

import POMDPs: Solver, Policy
import POMDPs: solve, action, value, update, initialize_belief, updater
import Base: ==, hash

export
    PruneSolver

include("vanilla.jl")

end # module
