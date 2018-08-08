
using Base.Test
using POMDPs, POMDPModels, POMDPToolbox
using IncrementalPruning
const IP = IncrementalPruning

@testset "Incremental Pruning Solver" begin
    @testset "Incremental Pruning Functions" begin
        # # dominate
        # return beleif state point where α dominates all other vectors in A
        α = [0.6, 0.6]
        A = Set([[1.0, -1.0], [0.0, 1.0]])
        x = IP.dominate(α, A)
        @test x[1] ≈ 0.667 atol = 0.001

        # filter arrays
        # return the set of non-dominated vectors
        A = Set([[1.0, -1.0], [0.0, 1.0], [0.3, 0.2], [0.9, 0.9]])
        B = Set([[1.0, -1.0], [0.0, 1.0], [0.9, 0.9]])
        Af = IP.filtervec(A)
        @test Af ⊆ B && B ⊆ Af # set equality

        # filter alpha vectors
        # return the set of non-dominated alpha vectors
        av1 = IP.AlphaVec([1.0, -1.0], 1)
        av2 = IP.AlphaVec([0.0, 1.0], 1)
        av3 = IP.AlphaVec([0.3, 0.2], 1)
        av4 = IP.AlphaVec([0.9, 0.9], 1)
        A = Set([av1,av2,av3,av4])
        B = Set([av1,av2,av4])
        Af = IP.filtervec(A)
        @test Af ⊆ B && B ⊆ Af # set equality

        # cross sum
        # return all cross-combinations of sums
        A = Set([IP.AlphaVec([1.0, -1.0], 1), IP.AlphaVec([0.0, 1.0], 1)])
        B = Set([IP.AlphaVec([10.0, 10.0], 1), IP.AlphaVec([20.0, 20.0], 1), IP.AlphaVec([30.0, 30.0], 1)])
        AB = IP.xsum(A, B)
        @test length(AB) == 6 # total number of elements
        @test Set([IP.AlphaVec(pop!(A).alpha + pop!(B).alpha, 1)]) ⊆ AB # arbitrary element is correct

        # incprune
        # return the filtered, iterated cross-sum of the inputs
        A = Set([[1.0, -1.0], [0.0, 1.0]])
        B = Set([[1.0, -2.0], [0.0, 1.0], [0.9, 0.9]])
        C = Set([[1.0, -3.0], [0.0, 1.0], [0.1, 0.9]])
        D = Set([[1.0, -4.0], [0.0, 1.0], [0.1, 0.9], [0.3, 0.3]])
        SZ = [A, B, C, D]
        SZref = IP.filtervec(IP.xsum(IP.xsum(IP.xsum(A,B),C),D))
        SZip = IP.incprune(SZ)
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
        @test IP.dpval(α,a,z,prob) == valref

        # dpupdate
        # return value unclear
        prob = BabyPOMDP()
        av1 = IP.AlphaVec([1.0, -1.0], 1)
        av2 = IP.AlphaVec([0.0, 1.0], 1)
        V0 = Set([av1, av2])
        @test length(IP.dpupdate(V0, prob)) == 3 # not sure why this is 3
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
        ns = n_states(pomdp)
        tpolicy = AlphaVectorPolicy(pomdp,[zeros(ns), ones(ns)])
        b = DiscreteBelief(pomdp, [0.8,0.2])
        @test action(tpolicy,b) == 1 # action "1" is optimal (action index = 2)

        # value
        # return value function at a belief
        pomdp = TigerPOMDP()
        ns = n_states(pomdp)
        tpolicy = AlphaVectorPolicy(pomdp,[[1.0, 0.0], [0.0, 1.0]])
        b = DiscreteBelief(pomdp, [0.8,0.2])
        b2 = DiscreteBelief(pomdp, [0.3,0.7])
        @test value(tpolicy, b) == 0.8 # α1 dominates
        @test value(tpolicy, b2) == 0.7 # α2 dominates

        # diff value
        # return maximum difference in value functions
        pomdp = TigerPOMDP()
        A = ordered_actions(pomdp)
        a1 = [4.0; 0.0]
        a2 = [0.0; 4.0]
        a3 = [-2.75; 2.75]
        a4 = [-2.8; 2.8]
        a5 = [2.8; 2.8]
        tX = [IP.AlphaVec(a1, A[1]); IP.AlphaVec(a2, A[2])]
        tY = [IP.AlphaVec(a3, A[3]), IP.AlphaVec(a4, A[3])]
        tZ = [IP.AlphaVec(a5, A[3])]
        @test IP.diffvalue(tY, tX, pomdp) ≈ 6.75 atol = 0.0001
        @test IP.diffvalue(tZ, tX, pomdp) ≈ 1.2 atol = 0.0001
    end
end
