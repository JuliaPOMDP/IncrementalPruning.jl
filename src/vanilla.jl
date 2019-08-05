############################################################
########### Incremental Pruning Solver #####################
############################################################

"""
    PruneSolver <: Solver

POMDP solver type using the incremental pruning algorithm.
"""
mutable struct PruneSolver <: Solver
    max_iterations::Int64
    tolerance::Float64
end

"""
    PruneSolver(; max_iterations, tolerance)

Initialize an incremental pruning solver with the `max_iterations` limit and desired `tolerance`.
"""
function PruneSolver(;max_iterations::Int64=10, tolerance::Float64=1e-3)
    return PruneSolver(max_iterations, tolerance)
end

"""
    AlphaVec

Alpha vector type of paired vector and action.
"""
struct AlphaVec
    alpha::Vector{Float64} # alpha vector
    action::Any # action associated wtih alpha vector
end

"""
    AlphaVec(vector, action_index)

Create alpha vector from `vector` and `action_index`.
"""
AlphaVec() = AlphaVec([0.0], 0)

# define alpha vector equality
==(a::AlphaVec, b::AlphaVec) = (a.alpha,a.action) == (b.alpha, b.action)
Base.hash(a::AlphaVec, h::UInt) = hash(a.alpha, hash(a.action, h))

"""
    create_policy(prune_solver, pomdp)

Create AlphaVectorPolicy for `prune_solver` using immediate rewards from `pomdp`.
"""
function create_policy(solver::PruneSolver, pomdp::POMDP)
    ns = n_states(pomdp)
    na = n_actions(pomdp)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    alphas = [[reward(pomdp,S[i],A[j]) for i in 1:ns] for j in 1:na]
    AlphaVectorPolicy(pomdp, alphas, A)
end

"""
    xsum(A,B)

Compute the cross-sum of alpha vectors `A` and `B`.

# Examples
```julia-repl
julia> a, b, c = AlphaVec([1.0, 2.0], 1), AlphaVec([3.0, 4.0], 1), AlphaVec([5.0, 6.0], 1)
julia> xsum(Set([a]), Set([b, c]))
Set(AlphaVec[AlphaVec([6.0, 8.0], 1), AlphaVec([4.0, 6.0], 1)])
```
"""
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

Compute the cross-sum of vectors `A` and `B`.

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

"""
    dominate(α, A)

The set of vectors in `A` dominated by `α`.
"""
function dominate(α::Array{Float64,1}, A::Set{Array{Float64,1}})
    ns = length(α)
    αset = Set{Array{Float64,1}}()
    push!(αset, α)
    Adiff = setdiff(A,αset)
    L = Model(JuMP.with_optimizer(Clp.Optimizer, LogLevel = 0))
    @variable(L, x[1:ns])
    @variable(L, δ)
    @objective(L, Max, δ)
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
    JuMP.optimize!(L)
    sol_status = JuMP.termination_status(L)
    if sol_status == :Infeasible
        return :Perp
        # println("Perp")
    else
        xval = JuMP.value.(x)
        dval = JuMP.value.(δ)
        if dval > 0
            return xval
            # println("xval")
        else
            return :Perp
            # println("Perp")
        end
    end
end # dominate

"""
    filtervec(F)

The set of vectors in `F` that contribute to the value function.
"""
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

"""
    filtervec(F)

The set of alpha vectors in `F` that contribute to the value function.
"""
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

"""
    incprune(Sz)

Standard incremental pruning of the set of vectors `Sz`.
"""
function incprune(SZ::Vector{Set{Vector{Float64}}})
    W = filtervec(xsum(SZ[1], SZ[2]))
    for i = 3:length(SZ)
        W = filtervec(xsum(W, SZ[i]))
    end
    W
end # incprune

"""
    dpval(α, a, z, pomdp)

Dynamic programming backup value of `α` for action `a` and observation `z` in `pomdp`.
"""
function dpval(α::Array{Float64,1}, a, z, prob::POMDP)
    S = ordered_states(prob)
    A = ordered_actions(prob)
    ns = n_states(prob)
    nz = n_observations(prob)
    γ = discount(prob)
    τ = Array{Float64,1}(undef, ns)
    for (sind,s) in enumerate(S)
        dist_t = transition(prob,s,a)
        exp_sum = 0.0
        for (spind, sp) in enumerate(S)
            dist_o = observation(prob,a,sp)
            pt = pdf(dist_t,sp)
            po = pdf(dist_o,z)
            exp_sum += α[spind] * po * pt
        end
        τ[sind] = (1 / nz) * reward(prob,s,a) + γ * exp_sum
    end
    τ
end

"""
    dpupdate(F, pomdp)

Dynamic programming update of `pomdp` for the set of alpha vectors `F`.
"""
function dpupdate(F::Set{AlphaVec}, prob::POMDP)
    alphas = [avec.alpha for avec in F]
    A = ordered_actions(prob)
    Z = ordered_observations(prob)
    na = n_actions(prob)
    nz = n_observations(prob)
    Sp = Set{AlphaVec}()
    # tcount = 0
    Sa = Set{AlphaVec}()
    for (aind, a) in enumerate(A)
        Sz = Vector{Set{Vector{Float64}}}(undef, nz)
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

"""
    diffvalue(Vnew, Vold, pomdp)

Maximum difference between new alpha vectors `Vnew` and old alpha vectors `Vold` in `pomdp`.
"""
function diffvalue(Vnew::Vector{AlphaVec},Vold::Vector{AlphaVec},pomdp::POMDP)
    ns = n_states(pomdp) # number of states in alpha vector
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    Anew = [avec.alpha for avec in Vnew]
    Aold = [avec.alpha for avec in Vold]
    dmax = -Inf # max difference
    for avecnew in Anew
        L = Model(JuMP.with_optimizer(Clp.Optimizer, LogLevel = 0))
        @variable(L, x[1:ns])
        @variable(L, t)
        @objective(L, Max, t)
        @constraint(L, x .>= 0)
        @constraint(L, x .<= 1)
        @constraint(L, sum(x) == 1)
        for avecold in Aold
            @constraint(L, (avecnew - avecold)' * x >= t)
        end
        JuMP.optimize!(L)
        dmax = max(dmax, JuMP.objective_value(L))
    end
    rmin = minimum(reward(pomdp,s,a) for s in S, a in A) # minimum reward
    if rmin < 0 # if negative rewards, find max difference from old to new
        for avecold in Aold
            L = Model(JuMP.with_optimizer(Clp.Optimizer, LogLevel = 0))
            @variable(L, x[1:ns])
            @variable(L, t)
            @objective(L, Max, t)
            @constraint(L, x .>= 0)
            @constraint(L, x .<= 1)
            @constraint(L, sum(x) == 1)
            for avecnew in Anew
                @constraint(L, (avecold - avecnew)' * x >= t)
            end
            JuMP.optimize!(L)
            dmax = max(dmax, JuMP.objective_value(L))
        end
    end
    dmax
end

"""
    solve(solver::PruneSolver, pomdp)

AlphaVectorPolicy for `pomdp` caluclated by the incremental pruning algorithm.
"""
function solve(solver::PruneSolver, prob::POMDP)
    # println("Solver started...")
    ϵ = solver.tolerance
    replimit = solver.max_iterations
    policy = create_policy(solver, prob)
    avecs = [AlphaVec(policy.alphas[i], policy.action_map[i]) for i in 1:length(policy.action_map)]
    Vold = Set(avecs)
    Vnew = Set{AlphaVec}()
    del = Inf
    reps = 0
    while del > ϵ && reps < replimit
        reps += 1
        Vnew = dpupdate(Vold, prob)
        del = diffvalue(collect(Vnew), collect(Vold), prob)
        Vold = Vnew
    end
    alphas_new = [v.alpha for v in Vnew]
    actions_new = [v.action for v in Vnew]
    policy = AlphaVectorPolicy(prob, alphas_new, actions_new)
    return policy
end

@POMDP_require solve(solver::PruneSolver, pomdp::POMDP) begin
    P = typeof(pomdp)
    S = state_type(P)
    A = action_type(P)
    @req discount(::P)
    @req n_states(::P)
    @req n_actions(::P)
    @subreq ordered_states(pomdp)
    @subreq ordered_actions(pomdp)
    @req transition(::P,::S,::A)
    @req reward(::P,::S,::A,::S)
    @req state_index(::P,::S)
    as = actions(pomdp)
    ss = states(pomdp)
    @req iterator(::typeof(as))
    @req iterator(::typeof(ss))
    s = first(iterator(ss))
    a = first(iterator(as))
    dist = transition(pomdp, s, a)
    D = typeof(dist)
    @req iterator(::D)
    @req pdf(::D,::S)
end
