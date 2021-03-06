export QuasiTrajectory,
       ndofs, 
       noptvars,
       fromarray!,
       fromarray,
       toarray!,
       toarray,
       dds!,
       tores

mutable struct QuasiTrajectory{N, DOMAIN, T}
     x::Matrix{T} # the data
     ω::Float64   # the frequency
    function QuasiTrajectory(x::Matrix{T}, ω::Real) where {T}
        if T <: Complex
            domain = :freq
            N = size(x, 2) - 1
            x[:, 1] .= real.(x[:, 1]) # imaginary mean mode must be zero
        else
            domain = :time
            N = div(size(x, 2) - 1, 2)
        end
        return new{N, domain, T}(x, ω)
    end
end
    
function QuasiTrajectory(M::Int, N::Int, ω::Real, domain::Symbol=:freq)
    if domain == :freq
       x = zeros(Complex{Float64}, M, N+1)
    elseif domain == :time
       x = zeros(M, 2*N+1)
    else
        throw(ArgumentError("invalid domain"))
    end
    return QuasiTrajectory(x, ω)
end

function tores(q̂::QuasiTrajectory{N, :freq}, P::Int) where {N}
    p̂ = QuasiTrajectory(ndofs(q̂), P, q̂.ω, :freq)
    p̂.x[:, 1:N+1] .= q̂.x
    return p̂
end

# number of degrees of freedom
ndofs(q::QuasiTrajectory) = size(q.x, 1)

# number of optimisation variables, including the frequency
noptvars(q̂::QuasiTrajectory{N}) where {N} = ndofs(q̂) * (2*N + 1) + 1


function squarednorm(q̂::QuasiTrajectory{N, :freq}) where {N}
    @views begin
        I = sum(abs2, q̂.x[:, 1])
        for k = 1:N
            I += 2 * sum(abs2, q̂.x[:, k+1])
        end
    end
    return I
end

function toarray!(w::Vector{Float64}, q̂::QuasiTrajectory{N, :freq}) where {N}
    # check size (including frequency)
    length(w) == noptvars(q̂) || throw(ArgumentError("invalid size"))

    # number of dofs
    M = ndofs(q̂)
    
    # add mean mode
    w[1:M] .= real.(q̂.x[:, 1])
    
    # then all others
    idx = M+1
    @inbounds for k = 1:N
        for j = 1:M
            w[idx]   = real.(q̂.x[j, k+1])
            w[idx+1] = imag.(q̂.x[j, k+1])
            idx += 2
        end
    end    
    
    # then the frequency
    w[end] = q̂.ω
    return w
end

toarray(q̂::QuasiTrajectory{N, :freq}) where {N} =
    toarray!(zeros(noptvars(q̂)), q̂)

function fromarray!(q̂::QuasiTrajectory{N, :freq}, w::Vector{Float64}) where {N}
    # check size (including frequency)
    length(w) == noptvars(q̂) || throw(ArgumentError("invalid size"))
    
    # number of modes
    M = ndofs(q̂)

    @inbounds begin 
        # add mean mode
        for j = 1:M
            q̂.x[j, 1] = w[j]
        end
    
        # then all others
        idx = M+1
        for k = 1:N, j = 1:M
            q̂.x[j, k+1] = w[idx] + im*w[idx+1]
            idx += 2
        end    
    
        q̂.ω = w[end]
    end
    return q̂
end

fromarray(w::AbstractVector, M::Int, N::Int) = 
    fromarray!(QuasiTrajectory(M, N, 0, :freq), w)

function dds!(ô::QuasiTrajectory{N, :freq}, q̂::QuasiTrajectory{N, :freq}) where {N}
    @inbounds for k in 0:N
        @views ô.x[:, k+1] .= im .* k .* q̂.x[:, k+1]
    end
    return ô
end