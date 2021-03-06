export lorenz!,
       lorenz_jac_adj!,
       lorenz_jac!

const σ = 10
const β = 8/3
const ρ = 28 

function lorenz!(f::AbstractVector, x::AbstractVector)
    _x, _y, _z = x
    @inbounds begin
        f[1] = σ * (_y - _x)
        f[2] = _x * (ρ - _z) - _y
        f[3] = _x*_y - β*_z
    end
    return f
end

function lorenz_jac_adj!(out::AbstractVector, x::AbstractVector, r::AbstractVector)
    _x, _y, _z = x
    @inbounds begin
        out[1] = - σ * r[1] + (ρ - _z) * r[2] + _y * r[3]
        out[2] =   σ * r[1] -        1 * r[2] + _x * r[3]
        out[3] =   0 * r[1] -       _x * r[2] -  β * r[3]
    end
    return out
end

function lorenz_jac!(out::AbstractVector, x::AbstractVector, r::AbstractVector)
    _x, _y, _z = x
    out[1] =      - σ * r[1] +  σ * r[2] +  0 * r[3]
    out[2] = (ρ - _z) * r[1] -  1 * r[2] - _x * r[3]
    out[3] =       _y * r[1] + _x * r[2] -  β * r[3]
    return out
end