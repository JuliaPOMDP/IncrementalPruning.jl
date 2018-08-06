############################################################
########### Incremental Pruning Solver #####################
############################################################

mutable struct PruneSolver <: Solver
    max_iterations::Int64
    tolerance::Float64
end

function PruneSolver(;max_iterations::Int64=10, tolerance::Float64=1e-3)
    return PruneSolver(max_iterations, tolerance)
end

# alpha vector struct to match alpha vectors to actions
struct AlphaVec
    alpha::Vector{Float64} # alpha vector
    action::Any # action associated wtih alpha vector
end

# alpha vector default constructor
AlphaVec() = AlphaVec([0.0], 0)

# define alpha vector equality
==(a::AlphaVec, b::AlphaVec) = (a.alpha,a.action) == (b.alpha, b.action)
Base.hash(a::AlphaVec, h::UInt) = hash(a.alpha, hash(a.action, h))

# policy struct
mutable struct PrunePolicy{P<:RPOMDP, A} <: Policy
    pomdp::P
    avecs::Vector{AlphaVec}
    alphas::Vector{Vector{Float64}}
    action_map::Vector{A}
end

# policy default constructor
function PrunePolicy(pomdp::RPOMDP)
    ns = n_states(pomdp)
    na = n_actions(pomdp)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    alphas = [[reward(pomdp,S[i],A[j]) for i in 1:ns] for j in 1:na]
    avecs = [AlphaVec(alphas[i], A[i]) for i in 1:na]
    action_map = A
    PrunePolicy(pomdp, avecs, alphas, action_map)
end

# policy with alphavec constructor
function PrunePolicy(pomdp::RPOMDP, avecs::Vector{AlphaVec})
    alphas = [avec.alpha for avec in avecs]
    action_map = [avec.action for avec in avecs]
    PrunePolicy(pomdp, avecs, alphas, action_map)
end

create_policy(solver::PruneSolver, pomdp::RPOMDP) = PrunePolicy(pomdp)

updater(p::PrunePolicy) = DiscreteUpdater(p.pomdp)

######## Incrmental Pruning Algorithm #########
# Accelerated Vector Pruning (Walraven and Spaan 2017)

# cross sum - alpha vectors
function xsum(A::Set{AlphaVec}, B::Set{AlphaVec})
    X = Set{AlphaVec}()
    for a in A, b in B
        @assert a.action == b.action "action mismatch"
        newVec = AlphaVec(a.alpha + b.alpha, a.action)
        push!(X, newVec)
    end
    X
end

"""
    xsum(A,B)

Compute the cross-sum of `A` and `B`.

# Examples
```julia-repl
julia> xsum(Set([[1.0, 2.0]]), Set([[3.0, 4.0], [5.0, 6.0]]))
Set(Array{Float64,1}[[4.0, 6.0], [6.0, 8.0]])
```
"""
function xsum(A::Set{Array{Float64,1}}, B::Set{Array{Float64,1}})
    X = Set{Array{Float64,1}}()
    for a in A, b in B
        push!(X, a + b)
    end
    X
end

# dominate(α,A)
# (Cassandra, Littman, Zhang 1997)
function dominate(α::Array{Float64,1}, A::Set{Array{Float64,1}})
    ns = length(α)
    αset = Set{Array{Float64,1}}()
    push!(αset, α)
    Adiff = setdiff(A,αset)
    L = Model(solver = ClpSolver())
    @variable(L, x[1:ns])
    @variable(L, δ)
    @objective(L, :Max, δ)
    @constraint(L, sum(x) == 1)
    @constraint(L, x[1:ns] .<= 1)
    @constraint(L, x[1:ns] .>= 0)
    # dx = size(x)
    # da = size(α)
    # println("Size of x: $dx")
    # println("Size of α: $da")
    for ap in Adiff
        # dap = size(ap)
        # println("Size of ap: $dap")
        @constraint(L, dot(x, α) >= δ + dot(x, ap))
    end
    sol = JuMP.solve(L)
    if sol == :Infeasible
        return :Perp
        # println("Perp")
    else
        xval = getvalue(x)
        dval = getvalue(δ)
        if dval > 0
            return xval
            # println("xval")
        else
            return :Perp
            # println("Perp")
        end
    end
end # dominate

# filtervec(F,S)
# (Cassandra, Littman, Zhang 1997)
function filtervec(F::Set{Array{Float64,1}})
    ns = length(sum(F))
    W = Set{Array{Float64,1}}()
    for i = 1: ns
        if !isempty(F)
            # println("i: $i  ")
            w = Array{Float64,1}()
            fsmax = -Inf
            for f in F
                # println("f: $f")
                if f[i] > fsmax
                    fsmax = f[i]
                    w = f
                    end
            end
            wset = Set{Array{Float64,1}}()
            push!(wset, w)
            # println("w: $w")
            push!(W,w)
            setdiff!(F,wset)
        end
    end
    while !isempty(F)
        # println("F Before: $F")
        ϕ = pop!(F)
        # println("F After: $F")
        # println("ϕ: $ϕ")
        # println("W: $W")
        x = dominate(ϕ, W)
        if x != :Perp
            push!(F, ϕ)
            w = Array{Float64,1}()
            fsmax = -Inf
            for f in F
                if dot(x, f) > fsmax
                    fsmax = dot(x, f)
                    w = f
                    end
            end
            wset = Set{Array{Float64,1}}()
            push!(wset, w)
            push!(W,w)
            setdiff!(F,wset)
        end
    end
    temp = [Float64[]]
    setdiff!(W,temp)
    W
end # filtervec

# filtervec for alpha vectors
function filtervec(A::Set{AlphaVec})
    ns = [length(av.alpha) for av in A][1]
    W = Set{AlphaVec}()
    for i = 1:ns
        if !isempty(A)
            # println("i: $i  ")
            w = AlphaVec()
            fsmax = -Inf
            for f in A
                # println("f: $f")
                if f.alpha[i] > fsmax
                    fsmax = f.alpha[i]
                    w = f
                    end
            end
            wset = Set{AlphaVec}()
            push!(wset, w)
            # println("w: $w")
            push!(W,w)
            setdiff!(A,wset)
        end
    end
    while !isempty(A)
        # println("F Before: $F")
        ϕ = pop!(A)
        ϕa = ϕ.alpha
        Wa = Set([w.alpha for w in W])
        # println("F After: $F")
        # println("ϕ: $ϕ")
        # println("W: $W")
        x = dominate(ϕa, Wa)
        if x != :Perp
            push!(A, ϕ)
            w = AlphaVec()
            fsmax = -Inf
            for f in A
                if dot(x, f.alpha) > fsmax
                    fsmax = dot(x, f.alpha)
                    w = f
                    end
            end
            wset = Set{AlphaVec}()
            push!(wset, w)
            push!(W,w)
            setdiff!(A,wset)
        end
    end
    W
end # filtervec

# incprune function
# (Cassandra, Littman, Zhang 1997)
function incprune(SZ::Array{Set{Array{Float64,1}},1})
    W = filtervec(xsum(SZ[1], SZ[2]))
    for i = 3:length(SZ)
        W = filtervec(xsum(W, SZ[i]))
    end
    W
end # incprune

# # dynamic programming backup value
# # (Cassandra, Littman, Zhang 1997)
# function dpval(α::Array{Float64,1}, a, z, prob::POMDP)
#     S = ordered_states(prob)
#     A = ordered_actions(prob)
#     ns = n_states(prob)
#     nz = n_observations(prob)
#     γ = discount(prob)
#     τ = Array{Float64,1}(ns)
#     for (sind,s) in enumerate(S)
#         dist_t = transition(prob,s,a)
#         exp_sum = 0.0
#         for (spind, sp) in enumerate(S)
#             dist_o = observation(prob,a,sp)
#             pt = pdf(dist_t,sp)
#             po = pdf(dist_o,z)
#             exp_sum += α[spind] * po * pt
#         end
#         τ[sind] = (1 / nz) * reward(prob,s,a) + γ * exp_sum
#     end
#     τ
# end

# Robust backup value
# dynamic programming backup value
# (Osogami 2015)
# (Cassandra, Littman, Zhang 1997)
function dpval(α::Array{Float64,1}, a, z, prob::RPOMDP)
    S = ordered_states(prob)
    A = ordered_actions(prob)
    ns = n_states(prob)
    nz = n_observations(prob)
    γ = discount(prob)
    τ = Array{Float64,1}(ns)
    for (sind,s) in enumerate(S)
        dist_t = transition(prob,s,a)
        exp_sum = 0.0
        for (spind, sp) in enumerate(S)
            uncset = RPOMDPs.observation(prob,a,sp)
            po_low = pdf(uncset.lower, obs_index(prob, z))
            po_hi = pdf(uncset.upper, obs_index(prob, z))
            pt = pdf(dist_t, sp)
            exp_sum += min(α[spind] * po_low * pt, α[spind] * po_hi * pt)
        end
        τ[sind] = (1 / nz) * reward(prob,s,a) + γ * exp_sum
    end
    τ
end

# dynamic programming update
# (Cassandra, Littman, Zhang 1997)
function dpupdate(F::Set{AlphaVec}, prob::RPOMDP)
    alphas = [avec.alpha for avec in F]
    A = ordered_actions(prob)
    Z = ordered_observations(prob)
    na = n_actions(prob)
    nz = n_observations(prob)
    Sp = Set{AlphaVec}()
    # tcount = 0
    Sa = Set{AlphaVec}()
    for (aind, a) in enumerate(A)
        Sz = Array{Set{Array{Float64,1}},1}(nz)
        for (zind, z) in enumerate(Z)
            # tcount += 1
            # println("DP Update Inner Loop: $tcount")
            V = Set(dpval(α,a,z,prob) for α in alphas)
            # println("V: $V")
            Sz[zind] = filtervec(V)
        end
        Sa = Set([AlphaVec(α,a) for α in incprune(Sz)])
        union!(Sp,Sa)
    end
    filtervec(Sp)
end

# Find maximum difference between new value function and old value function
function diffvalue(Vnew::Array{AlphaVec,1},Vold::Array{AlphaVec,1},pomdp::RPOMDP)
    ns = n_states(pomdp) # number of states in alpha vector
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    Anew = [avec.alpha for avec in Vnew]
    Aold = [avec.alpha for avec in Vold]
    dmax = -Inf # max difference
    for avecnew in Anew
        L = Model(solver = ClpSolver())
        @variable(L, x[1:ns])
        @variable(L, t)
        @objective(L, :Max, t)
        @constraint(L, x .>= 0)
        @constraint(L, x .<= 1)
        @constraint(L, sum(x) == 1)
        for avecold in Aold
            @constraint(L, (avecnew - avecold)' * x >= t)
        end
        sol = JuMP.solve(L)
        dmax = max(dmax, getobjectivevalue(L))
    end
    rmin = minimum(reward(pomdp,s,a) for s in S, a in A) # minimum reward
    if rmin < 0 # if negative rewards, find max difference from old to new
        for avecold in Aold
            L = Model(solver = ClpSolver())
            @variable(L, x[1:ns])
            @variable(L, t)
            @objective(L, :Max, t)
            @constraint(L, x .>= 0)
            @constraint(L, x .<= 1)
            @constraint(L, sum(x) == 1)
            for avecnew in Anew
                @constraint(L, (avecold - avecnew)' * x >= t)
            end
            sol = JuMP.solve(L)
            dmax = max(dmax, getobjectivevalue(L))
        end
    end
    dmax
end

# solve POMDP with incremental pruning
function solve(solver::PruneSolver, prob::RPOMDP)
    # println("Solver started...")
    ϵ = solver.tolerance
    replimit = solver.max_iterations
    policy = PrunePolicy(prob)
    Vold = Set(policy.avecs)
    Vnew = Set{AlphaVec}()
    del = Inf
    reps = 0
    while del > ϵ && reps < replimit
        reps += 1
        Vnew = dpupdate(Vold, prob)
        del = diffvalue(collect(Vnew), collect(Vold), prob)
        Vold = Vnew
    end
    policy = PrunePolicy(prob, collect(Vnew))
    return policy
end

alphas(policy::PrunePolicy) = policy.alphas

function action(policy::PrunePolicy, b::DiscreteBelief)
    alphas = policy.alphas
    nvec = length(alphas) # number of alpha vectors
    ns = length(alphas[1]) # number of states in first alpha vector
    @assert length(b.b) == ns "Length of belief and alpha-vector size mismatch"
    util = [alphas[i]' * b.b for i = 1:nvec] # utility for each α-vector
    imax = indmax(util)
    return policy.action_map[imax] # action associated with max utility for b
end


function value(policy::PrunePolicy, b::DiscreteBelief)
    alphas = policy.alphas
    nvec = length(alphas) # number of alpha vectors
    ns = length(alphas[1]) # number of states in first alpha vector
    @assert length(b.b) == ns "Length of belief and alpha-vector size mismatch"
    util = [alphas[i]' * b.b for i = 1:nvec] # utility for each α-vector
    return maximum(util)
end

function value(policy::PrunePolicy, b)
    if isa(b, state_type(policy.pomdp))
        return state_value(policy, b)
    end
    return value(policy, DiscreteBelief(belief_vector(policy, b)))
end

function state_value(policy::PrunePolicy, s)
    si = state_index(policy.pomdp, s)
    maximum(alpha[si] for alpha in policy.alphas)
end

function action(policy::PrunePolicy, b)
    return action(policy, DiscreteBelief(belief_vector(policy, b)))
end

function belief_vector(policy::PrunePolicy, b)
    bv = Array{Float64}(n_states(policy.pomdp))
    for (i,s) in enumerate(ordered_states(policy.pomdp))
        bv[i] = pdf(b, s)
    end
    return bv
end

function unnormalized_util(policy::PrunePolicy, b::AbstractParticleBelief)
    util = zeros(n_actions(policy.pomdp))
    for (i, s) in enumerate(particles(b))
        si = state_index(policy.pomdp, s)
        as = [alpha[si] for alpha in policy.alphas]
        util += weight(b, i) * as
    end
    return util
end

function action(policy::PrunePolicy, b::AbstractParticleBelief)
    util = unnormalized_util(policy, b)
    imax = indmax(util)
    return policy.action_map[imax]
end

value(policy::PrunePolicy, b::AbstractParticleBelief) = maximum(unnormalized_util(policy, b))/weight_sum(b)

# @POMDP_require solve(solver::PruneSolver, pomdp::RPOMDP) begin
#     P = typeof(pomdp)
#     S = state_type(P)
#     A = action_type(P)
#     @req discount(::P)
#     @req n_states(::P)
#     @req n_actions(::P)
#     @subreq ordered_states(pomdp)
#     @subreq ordered_actions(pomdp)
#     @req transition(::P,::S,::A)
#     @req reward(::P,::S,::A,::S)
#     @req state_index(::P,::S)
#     as = actions(pomdp)
#     ss = states(pomdp)
#     @req iterator(::typeof(as))
#     @req iterator(::typeof(ss))
#     s = first(iterator(ss))
#     a = first(iterator(as))
#     dist = transition(pomdp, s, a)
#     D = typeof(dist)
#     @req iterator(::D)
#     @req pdf(::D,::S)
# end
