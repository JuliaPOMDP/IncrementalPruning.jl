
using Base.Test
push!(LOAD_PATH,"C:\\Users\\op\\Documents\\Julia Projects\\IncrementalPruning.jl\\src")
using POMDPs, POMDPModels, POMDPToolbox
using IncrementalPruning

@testset "Incremental Pruning Solver" begin
    @testset "Incremental Pruning Functions" begin
        # # dominate
        # return beleif state point where α dominates all other vectors in A
        α = [0.6, 0.6]
        A = Set([[1.0, -1.0], [0.0, 1.0]])
        x = dominate(α, A)
        @test x[1] ≈ 0.667 atol = 0.001

        # filter arrays
        # return the set of non-dominated vectors
        A = Set([[1.0, -1.0], [0.0, 1.0], [0.3, 0.2], [0.9, 0.9]])
        B = Set([[1.0, -1.0], [0.0, 1.0], [0.9, 0.9]])
        Af = filtervec(A)
        @test Af ⊆ B && B ⊆ Af # set equality

        # filter alpha vectors
        # return the set of non-dominated alpha vectors
        av1 = AlphaVec([1.0, -1.0], 1)
        av2 = AlphaVec([0.0, 1.0], 1)
        av3 = AlphaVec([0.3, 0.2], 1)
        av4 = AlphaVec([0.9, 0.9], 1)
        A = Set([av1,av2,av3,av4])
        B = Set([av1,av2,av4])
        Af = filtervec(A)
        @test Af ⊆ B && B ⊆ Af # set equality

        # cross sum
        # return all cross-combinations of sums
        A = Set([AlphaVec([1.0, -1.0], 1), AlphaVec([0.0, 1.0], 1)])
        B = Set([AlphaVec([10.0, 10.0], 1), AlphaVec([20.0, 20.0], 1), AlphaVec([30.0, 30.0], 1)])
        AB = xsum(A, B)
        @test length(AB) == 6 # total number of elements
        @test Set([AlphaVec(pop!(A).alpha + pop!(B).alpha, 1)]) ⊆ AB # arbitrary element is correct

        # incprune
        # return the filtered, iterated cross-sum of the inputs
        A = Set([[1.0, -1.0], [0.0, 1.0]])
        B = Set([[1.0, -2.0], [0.0, 1.0], [0.9, 0.9]])
        C = Set([[1.0, -3.0], [0.0, 1.0], [0.1, 0.9]])
        D = Set([[1.0, -4.0], [0.0, 1.0], [0.1, 0.9], [0.3, 0.3]])
        SZ = [A, B, C, D]
        SZref = filtervec(xsum(xsum(xsum(A,B),C),D))
        SZip = incprune(SZ)
        @test SZip ⊆ SZref && SZref ⊆ SZip # set equality

        # dpval
        # return a vector with the value of each state
        prob = BabyPOMDP()
        A = ordered_actions(prob)
        Z = ordered_observations(prob)
        a = A[1] # feed
        z = Z[2] # cry
        α = [-10.0, 0.0] # reward for: [hungry, full]
        valref = [-7.5, -2.5] # N-1 step value of S = [hungry, full]
        @test dpval(α,a,z,prob) == valref

        # dpupdate
        # return value may be incorrect
        prob = BabyPOMDP()
        av1 = AlphaVec([1.0, -1.0], 1)
        av2 = AlphaVec([0.0, 1.0], 1)
        V0 = Set([av1, av2])
        @test length(dpupdate(V0, prob)) == 3 # not sure if 3 is actually right
    end

    @testset "Solver Functions" begin
        # PruneSolver
        # return simulated expected total discounted reward
        solver = PruneSolver()
        @test test_solver(solver, TigerPOMDP()) ≈ 17.71 atol = 0.01
        @test test_solver(solver, BabyPOMDP()) ≈ -17.96 atol = 0.01

        # action
        # return best action at a belief
        pomdp = TigerPOMDP()
        A = ordered_actions(pomdp)
        ns = n_states(pomdp)
        talphavec = AlphaVec(zeros(ns), A[1])
        talphavec2 = AlphaVec(ones(ns), A[2])
        tpolicy = PrunePolicy(pomdp,[talphavec, talphavec2])
        b = DiscreteBelief([0.8,0.2])
        @test action(tpolicy,b) == 1 # action "1" is optimal (action index = 2)

        # value
        # return value function at a belief
        pomdp = TigerPOMDP()
        A = ordered_actions(pomdp)
        ns = n_states(pomdp)
        talphavec = AlphaVec([1.0, 0.0], A[1])
        talphavec2 = AlphaVec([0.0, 1.0], A[2])
        tpolicy = PrunePolicy(pomdp,[talphavec, talphavec2])
        b = DiscreteBelief([0.8,0.2])
        b2 = DiscreteBelief([0.3,0.7])
        @test value(tpolicy, b) == 0.8 # α1 dominates
        @test value(tpolicy, b2) == 0.7 # α2 dominates

        # state value
        # return value function at a state
        pomdp = TigerPOMDP()
        A = ordered_actions(pomdp)
        ns = n_states(pomdp)
        talphavec = AlphaVec([1.0, 0.5], A[1])
        talphavec2 = AlphaVec([0.0, 2.0], A[2])
        tpolicy = PrunePolicy(pomdp,[talphavec, talphavec2])
        @test IncrementalPruning.state_value(tpolicy, false) == 1.0 # α1 dominates
        @test IncrementalPruning.state_value(tpolicy, true) == 2.0 # α2 dominates

        # diff value
        # return maximum difference in value functions
        pomdp = TigerPOMDP()
        a1 = [4.0; 0.0]
        a2 = [0.0; 4.0]
        a3 = [-2.75; 2.75]
        a4 = [-2.8; 2.8]
        a5 = [2.8; 2.8]
        tX = [AlphaVec(a1, A[1]); AlphaVec(a2, A[2])]
        tY = [AlphaVec(a3, A[3]), AlphaVec(a4, A[3])]
        tZ = [AlphaVec(a5, A[3])]
        @test diffvalue(tY, tX, pomdp) ≈ 6.75 atol = 0.0001
        @test diffvalue(tZ, tX, pomdp) ≈ 1.2 atol = 0.0001
    end
end
