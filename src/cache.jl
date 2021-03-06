export Cache,
       globalresidual,
       globalresidualgrad

struct Cache{IP, FP, TMPF, TMPT, S, A}
          iplan::IP   # inverse plan
          fplan::FP   # forward plan
          tmp_f::TMPF # freq domain temporary objects
          tmp_t::TMPT # time domain temporary objects
            sys::S    # right hand side of system
    sys_jac_adj::A    # adjoint jacobian op
    function Cache(M::Int, N::Int, sys, sys_jac_adj)
        fplan = FFTPLAN(M, N)
        iplan = IFFTPLAN(M, N)
        tmp_f = ntuple(i->QuasiTrajectory(M, N, 0, :freq), 8)
        tmp_t = ntuple(i->QuasiTrajectory(M, N, 0, :time), 4)
        args = (iplan, fplan, tmp_f, tmp_t, sys, sys_jac_adj)
        return new{typeof(args[1]), 
                   typeof(args[2]), 
                   typeof(args[3]), 
                   typeof(args[4]), 
                   typeof(args[5]), 
                   typeof(args[6])}(args...)
    end
end


function _residual!(r̂::QuasiTrajectory{N, :freq}, 
                    q̂::QuasiTrajectory{N, :freq}, 
                cache::Cache) where {N}
    # extract aliases
    dq̂  = cache.tmp_f[1]
    f̂   = cache.tmp_f[2]
    q   = cache.tmp_t[1]
    f   = cache.tmp_t[2]
    sys = cache.sys
    
    # compute loop derivative
    dds!(dq̂, q̂)
    
    # go to time domain
    LinearAlgebra.mul!(q, cache.iplan, q̂)
    
    # compute nonlinearity in time domain
    @inbounds for i = 1:(2N+1)
        @views sys(f.x[:, i], q.x[:, i])
    end
    
    # go back to freq domain
    LinearAlgebra.mul!(f̂, cache.fplan, f)
    
    # compute residual in freq domain
    r̂.x .= q̂.ω .* dq̂.x .- f̂.x
    
    # set frequency
    r̂.ω = q̂.ω
    
    return r̂
end

function globalresidual(w::AbstractVector, cache::Cache)
    q̂ = cache.tmp_f[3]
    r̂ = cache.tmp_f[4]
    fromarray!(q̂, w)
    _residual!(r̂, q̂, cache)
    return 0.5 * squarednorm(r̂)
end

function _residualgrad!(dRdq̂::QuasiTrajectory{N, :freq}, 
                           q̂::QuasiTrajectory{N, :freq}, 
                       cache::Cache) where {N}
    # extract aliases
    dq̂  = cache.tmp_f[3]
    r̂   = cache.tmp_f[4]
    dr̂  = cache.tmp_f[5]
    p̂   = cache.tmp_f[6]
    q   = cache.tmp_t[1] # this is computed in _residual!
    r   = cache.tmp_t[3]
    p   = cache.tmp_t[4]
    sys_jac_adj = cache.sys_jac_adj
    
    # compute residual
    _residual!(r̂, q̂, cache) # this also computes `q` in cache.tmp_t[1]
    
    # go to residual in time domain
    LinearAlgebra.mul!(r, cache.iplan, r̂)
    
    # compute action of adjoint jacobian in time domain
    @inbounds for i = 1:(2N+1)
        @views sys_jac_adj(p.x[:, i], q.x[:, i], r.x[:, i])
    end
    
    # compute loop derivative of residual
    dds!(dr̂, r̂)
    
    # go back to freq domain
    LinearAlgebra.mul!(p̂, cache.fplan, p)
    
    # compute residual gradient in freq domain
    dRdq̂.x .= .- q̂.ω .* dr̂.x .- p̂.x
    
    # set gradient wrt to frequency TODO
    dRdq̂.ω = 0
    
    return 0.5 * squarednorm(r̂), dRdq̂
end

function globalresidualgrad(w::AbstractVector, grad::AbstractVector, cache::Cache)
    # aliases
    q̂    = cache.tmp_f[7]
    dRdq̂ = cache.tmp_f[8]
    
    # get trajectory
    fromarray!(q̂, w)
    
    # compute global residual and its gradient 
    ret, dRdq̂ = _residualgrad!(dRdq̂, q̂, cache)
    
    # put residual into array form
    toarray!(grad, dRdq̂)
    
    return ret
end