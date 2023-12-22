module IncrementalPruning

using LinearAlgebra
using POMDPs
using POMDPLinter: @POMDP_require
using JuMP, GLPK, MathOptInterface
using POMDPTools
using Printf

import POMDPs: Solver, Policy
import POMDPs: solve, action, value, update, initialize_belief, updater
import Base: ==, hash

export
    PruneSolver

include("vanilla.jl")

end # module
