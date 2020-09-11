module IncrementalPruning

using LinearAlgebra
using POMDPs, POMDPPolicies
using POMDPLinter: @POMDP_require
using JuMP, GLPK
using POMDPModelTools: ordered_states, StateActionReward

import POMDPs: Solver, Policy
import POMDPs: solve, action, value, update, initialize_belief, updater
import Base: ==, hash

export
    PruneSolver

include("vanilla.jl")

end # module
