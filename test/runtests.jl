using LinearAlgebra
using Resolver
using Test

@testset "fromtoarray                            " begin
    M, N = 4, 20
    â = QuasiTrajectory(M, N, 1.234, :freq)
    ĉ = QuasiTrajectory(M, N, 3.456, :freq)
    w = zeros(noptvars(â))
    fromarray!(ĉ, toarray!(w, â))
    @test norm(â.x .- ĉ.x) == 0
    @test toarray(â) == toarray!(w, â)

    @allocated fromarray!(ĉ, toarray!(w, â)) == 0
end


@testset "fftplans                               " begin
    M, N = 2, 3
    â = QuasiTrajectory(M, N, 1.234, :freq)
    ĉ = QuasiTrajectory(M, N, 1.234, :freq)
    a = QuasiTrajectory(M, N, 1.234, :time)
    
    iplan = IFFTPLAN(M, N)
    fplan =  FFTPLAN(M, N)

    mul!(a, iplan, â)
    mul!(ĉ, fplan, a)
    @test norm(â.x .- ĉ.x) < 1e-14
end

@testset "lorenz                                 " begin
    _β, _σ, _ρ = Resolver.β, Resolver.σ, Resolver.ρ
    @test norm(lorenz!(zeros(3), [0, 0, 0])) < 1e-13
    @test norm(lorenz!(zeros(3), [sqrt(_β*(_ρ-1)), sqrt(_β*(_ρ-1)), _ρ-1])) < 1e-13

    # check adjoint operator
    @test (([7, 8, 9]' * lorenz_jac_adj!(zeros(3), [1, 2, 3], [4, 5, 6])) - 
           ([4, 5, 6]' * lorenz_jac!(    zeros(3), [1, 2, 3], [7, 8, 9]))) == 0
end

@testset "cache                                  " begin
    # just check the constructor is OK
    M, N = 3, 4
    cache = Cache(M, N, lorenz!, lorenz_jac_adj!)
end