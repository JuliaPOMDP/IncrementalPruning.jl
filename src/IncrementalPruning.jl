module IncrementalPruning

using RPOMDPs
using RPOMDPToolbox
using ParticleFilters

using JuMP, Clp

import RPOMDPs: Solver, Policy
import RPOMDPs: solve, action, value, update, initialize_belief, updater
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

include("incprune.jl")
include("valueiteration.jl")

end # module
