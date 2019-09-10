module IncrementalPruning

using LinearAlgebra
using POMDPs, POMDPModelTools, POMDPPolicies
using ParticleFilters
using JuMP, GLPK, MathOptInterface

import POMDPs: Solver, Policy
import POMDPs: solve, action, value, update, initialize_belief, updater
import Base: ==, hash

export
    PruneSolver,
    solve

include("vanilla.jl")

end # module
