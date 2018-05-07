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
    PrunePolicy,
    PruneUpdater,
    AlphaVec,
    diffvalue,
    solve,
    action,
    value,
    state_value,
    update,
    initialize_belief,
    updater,
    create_policy,
    create_belief,
    dominate,
    filtervec,
    incprune,
    dpupdate,
    dpval,
    xsum

include("vanilla.jl")

end # module
