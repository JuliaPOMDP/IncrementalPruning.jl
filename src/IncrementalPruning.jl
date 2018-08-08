module IncrementalPruning

using POMDPs
using POMDPToolbox
using ParticleFilters

using JuMP, Clp

import POMDPs: Solver, Policy
import POMDPs: solve, action, value, update, initialize_belief, updater
import Base: ==, hash

export
    PruneSolver,
    solve

include("vanilla.jl")

end # module
