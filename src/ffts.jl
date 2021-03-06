import FFTW
import LinearAlgebra

export FFTPLAN,
       IFFTPLAN,
       FFT,
       IFFT

const _fftargs = (flags=FFTW.EXHAUSTIVE, timelimit=10)

struct FFTPLAN{N, P}
    plan::P
    function FFTPLAN(M::Int, N::Int)
        plan = FFTW.plan_rfft(zeros(M, 2*N+1), [2])#; _fftargs...)
        return new{N, typeof(plan)}(plan)
    end
end

function LinearAlgebra.mul!(out::QuasiTrajectory{N, :freq},
                              p::FFTPLAN{N},
                              q::QuasiTrajectory{N, :time}) where {N}
    LinearAlgebra.mul!(out.x, p.plan, q.x)
    out.x ./= 2*N + 1
    return out
end

FFT(q::QuasiTrajectory{N, :time}) where {N} =
    mul!(QuasiTrajectory(ndofs(q), N, 0.0, :freq), FFTPLAN(ndofs(q), N), q)

struct IFFTPLAN{N, P}
    plan::P
    function IFFTPLAN(M::Int, N::Int)
        plan = FFTW.plan_brfft(zeros(Complex{Float64}, M, N+1), 2*N+1, [2])#; _fftargs...)
        return new{N, typeof(plan)}(plan)
    end
end

function LinearAlgebra.mul!(out::QuasiTrajectory{N, :time},
                              p::IFFTPLAN{N},
                              q::QuasiTrajectory{N, :freq}) where {N}
    LinearAlgebra.mul!(out.x, p.plan, q.x)
    return out
end

IFFT(q̂::QuasiTrajectory{N, :freq}) where {N} =
    LinearAlgebra.mul!(QuasiTrajectory(ndofs(q̂), N, 0.0, :time), IFFTPLAN(ndofs(q̂), N), q̂)