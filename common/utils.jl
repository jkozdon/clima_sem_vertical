import GaussQuadrature
using LinearAlgebra: Diagonal, I, eigvals
using SparseArrays: sparse

"""
    lglpoints(::Type{T}, N::Integer) where T <: AbstractFloat

returns the points `r` and weights `w` associated with the `N+1`-point
Gauss-Legendre-Lobatto quadrature rule of type `T`

"""
function lglpoints(::Type{T}, N::Integer) where {T <: AbstractFloat}
    @assert N ≥ 1
    GaussQuadrature.legendre(T, N + 1, GaussQuadrature.both)
end

"""
    baryweights(r)

returns the barycentric weights associated with the array of points `r`

Reference:
  [Berrut2004](@cite)
"""
function baryweights(r::AbstractVector{T}) where {T}
    Np = length(r)
    wb = ones(T, Np)

    for j in 1:Np
        for i in 1:Np
            if i != j
                wb[j] = wb[j] * (r[j] - r[i])
            end
        end
        wb[j] = T(1) / wb[j]
    end
    wb
end

"""
    spectralderivative(r::AbstractVector{T},
                       wb=baryweights(r)::AbstractVector{T}) where T

returns the spectral differentiation matrix for a polynomial defined on the
points `r` with associated barycentric weights `wb`

Reference:
 - [Berrut2004](@cite)
"""
function spectralderivative(
    r::AbstractVector{T},
    wb = baryweights(r)::AbstractVector{T},
) where {T}
    Np = length(r)
    @assert Np == length(wb)
    D = zeros(T, Np, Np)

    for k in 1:Np
        for j in 1:Np
            if k == j
                for l in 1:Np
                    if l != k
                        D[j, k] = D[j, k] + T(1) / (r[k] - r[l])
                    end
                end
            else
                D[j, k] = (wb[k] / wb[j]) / (r[j] - r[k])
            end
        end
    end
    D
end

"""
    interpolationmatrix(rsrc::AbstractVector{T}, rdst::AbstractVector{T},
                        wbsrc=baryweights(rsrc)::AbstractVector{T}) where T

returns the polynomial interpolation matrix for interpolating between the points
`rsrc` (with associated barycentric weights `wbsrc`) and `rdst`

Reference:
 - [Berrut2004](@cite)
"""
function interpolationmatrix(
    rsrc::AbstractVector{T},
    rdst::AbstractVector{T},
    wbsrc = baryweights(rsrc)::AbstractVector{T},
) where {T}
    Npdst = length(rdst)
    Npsrc = length(rsrc)
    @assert Npsrc == length(wbsrc)
    I = zeros(T, Npdst, Npsrc)
    for k in 1:Npdst
        for j in 1:Npsrc
            I[k, j] = wbsrc[j] / (rdst[k] - rsrc[j])
            if !isfinite(I[k, j])
                I[k, :] .= T(0)
                I[k, j] = T(1)
                break
            end
        end
        d = sum(I[k, :])
        I[k, :] = I[k, :] / d
    end
    I
end

"""
    DG_cutoff_filter_matrix(r, Nc)

Returns the filter matrix that takes function values at the interpolation
`N+1` points, `r`, and zeros out modes `Nc:N`
"""
function DG_cutoff_filter_matrix(r, Nc)
    N = length(r) - 1
    T = eltype(r)

    @assert N > 0

    a, b = GaussQuadrature.legendre_coefs(T, N)
    V = GaussQuadrature.orthonormal_poly(r, a, b)

    Σ = ones(Int, N + 1)
    Σ[(Nc:N) .+ 1] .= 0

    V * Diagonal(Σ) / V
end

"""
    CG_cutoff_filter_matrix(r, rc)

Returns the filter matrix that takes function values at the interpolation
`N+1` points, `r`, and L2 projects it to the `Nc + 1` points, `rc`, preseving
the end point values
"""
function CG_cutoff_filter_matrix(r, rc)
    T = eltype(r)

    N = length(r) - 1
    @assert N > 0

    Nc = length(rc) - 1
    @assert Nc < N

    # Mass matrix for full space
    V = GaussQuadrature.orthonormal_poly(r,
               GaussQuadrature.legendre_coefs(T, N)...)
    M = I / (V * V')

    # Mass matrix for reduced space
    a, b = GaussQuadrature.legendre_coefs(T, Nc)
    Vc = GaussQuadrature.orthonormal_poly(rc,
               GaussQuadrature.legendre_coefs(T, Nc)...)
    Mc = I / (Vc * Vc')

    # Reduced to full interpolation
    Ic = interpolationmatrix(rc, r)

    # First define a projection that assumes zero end values
    P = zeros(T, Nc + 1, N + 1)
    P[2:Nc, :] += (Mc[2:Nc, 2:Nc] \ Ic[2:N, 2:Nc]' * M[2:N, :])

    # get the linear end point part
    lin = zeros(T, N + 1, N + 1)
    lin[:, [1, N + 1]] += interpolationmatrix(T.([-1, 1]), r)

    linc = zeros(T, Nc + 1, N + 1)
    linc[:, [1, N + 1]] += interpolationmatrix(T.([-1, 1]), rc)

    # add linear part and project after removing linear part
    P = linc + P * (I - lin)
end
 
function dss!(x)
  x[1, 2:end] .= x[end, 1:end-1] .= (x[1, 2:end] .+ x[end, 1:end-1])
  x
end

function dg_to_cg!(x, M_dg, Q_M_cg)
  x .= M_dg .* x
  dss!(x)
  x .= x ./ Q_M_cg
  x
end

function scatter_matrix(N, K)
  Nq   = N + 1
  N_dg = K * Nq
  Is = 1:N_dg
  Js = zeros(Int, Nq, K)
  Js[1:N, :] .= reshape(1:(N*K), N, K)
  Js[Nq, K] = N * K + 1
  Js[Nq, 1:K-1] .= Js[1, 2:K]
  Q = sparse(Is, Js[:], ones(Int, length(Is)))
end

function timestep!(q, f!, dt, (t0, t1))
  T = eltype(q[1])

  RKA = (
         T(0),
         T(-567301805773 // 1357537059087),
         T(-2404267990393 // 2016746695238),
         T(-3550918686646 // 2091501179385),
         T(-1275806237668 // 842570457699),
  )

  RKB = (
         T(1432997174477 // 9575080441755),
         T(5161836677717 // 13612068292357),
         T(1720146321549 // 2090206949498),
         T(3134564353537 // 4481467310338),
         T(2277821191437 // 14882151754819),
  )

  RKC = (
         T(0),
         T(1432997174477 // 9575080441755),
         T(2526269341429 // 6820363962896),
         T(2006345519317 // 3224310063776),
         T(2802321613138 // 2924317926251),
  )

  Δq = ntuple(i->fill!(similar(q[i]), 0), length(q))
  Δq_ = ntuple(i->fill!(similar(q[i]), 0), length(q))
  if q isa NamedTuple
    Δq_ = NamedTuple{keys(q)}(Δq_)
  end

  nstep = ceil(Int, (t1 - t0) / dt)
  dt = (t1 - t0) / nstep
  for step = 1:nstep
    t = t0 + (step - 1) * dt
    # if mod(step, 1000) == 0
    #   println((t, t1))
    #   for i = 1:length(q)
    #     println((i, extrema(q[i])))
    #   end
    # end
    for s in 1:length(RKA)
      f!(Δq_, q, t + RKC[s] * dt)
      for i = 1:length(q)
        Δq[i] .+= Δq_[i]
        Δq_[i] .= 0
        q[i] .+= RKB[s] * dt * Δq[i]
        Δq[i] .*= RKA[s % length(RKA) + 1]
      end
    end
  end

  nothing
end

#=
function ten!(∂q, q, t)
  # y1 = sin(t)
  # y2 = exp(-t)
  # y3 = 0
  ∂q[1] .+= -q[1] .+ sin(t) .+ cos(t)
  ∂q[2] .+= -q[2]
  ∂q[3] .+= -q[3] - q[2] .+ exp(-t) + q[1] .- sin(t)
end
dts = 2.0 .^ (0:-1:-8)
error = zeros(length(dts))
for (i, dt) = enumerate(dts)
  q = ([0.0],[1.0], [0.0])
  timestep!(q, ten!, dt, (0.0, 1.0))
  error[i] = abs(q[1][1] - sin(1)) + abs(q[2][1] - exp(-1)) + abs(q[3][1])
end
rate = (log.(error[1:end-1]) - log.(error[2:end])) ./ (log.(dts[1:end-1]) - log.(dts[2:end]))
=#
