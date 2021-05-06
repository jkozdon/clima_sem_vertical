include("cg_sem.jl")
using StaticArrays: SVector

wave_speed(ρ, p, γ) = sqrt(γ * p ./ ρ)

Base.@kwdef struct Params{T}
    T_virt_surf::T = 280
    T_min_ref::T = 230
    MSLP::T = 10^5
    grav::T = 98 // 100
    R_d::T = 287
    γ::T = 7 // 5
end

# Taken from Daniel's code
function decaying_temperature_profile(params::Params, z)
    # Scale height for surface temperature
    H_sfc = params.R_d * params.T_virt_surf / params.grav
    H_t = H_sfc

    z′ = z / H_t
    tanh_z′ = tanh(z′)

    ΔTv = params.T_virt_surf - params.T_min_ref
    Tv = params.T_virt_surf - ΔTv * tanh_z′

    ΔTv′ = ΔTv / params.T_virt_surf
    p = -H_t * (z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
    p /= H_sfc * (1 - ΔTv′^2)
    p = params.MSLP * exp(p)
    ρ = p / (params.R_d * Tv)
    return (p = p, ρ = ρ)
end

function element_split_form!(∂q, q, O, H, z)
  Nq, K, nflds = size(q)

  Q = O.Q
  for e in 1:K
    for i in 1:Nq, j in 1:Nq
      # get the two states
      qi = @view q[i, e, :]
      qj = @view q[j, e, :]
      zi = z[i, e]
      zj = z[j, e]
      # Evaluate the flux difference form:
      #   ((H ∘ Qᵀ) - (Q ∘ H)) 𝟙
      ∂q[i, e, :] += (Q[i, j] - Q[j, i]) * H(qi, qj, zi, zj)
    end
  end
end

pressure(ρ, w, ρe, z, g, γ) = (γ - 1) * (ρe - ρ * w^2 / 2 - ρ * g * z)

function KennedyGruber(q_i, q_j, z_i, z_j, grav, pressure)
  (ρ_i, w_i, ρe_i) = q_i[1:3]
  (ρ_j, w_j, ρe_j) = q_j[1:3]

  e_i, e_j = ρe_i / ρ_i, ρe_j / ρ_j

  ρ̄ = (ρ_i + ρ_j) / 2
  w̄ = (w_i + w_j) / 2
  Δw = (w_j - w_i) / 2
  ē = (e_i + e_j) / 2
  Δϕ = grav * (z_j - z_i) / 2

  p_i = pressure(ρ_i, w_i, ρe_i, z_i, grav)
  p_j = pressure(ρ_j, w_j, ρe_j, z_j, grav)
  p̄ = (p_i + p_j) / 2

  fρ  = ρ̄ * w̄
  fρw = ρ̄ * w̄ * Δw + p̄ + ρ̄ * Δϕ
  fρe = (ρ̄ * ē + p̄) * w̄
  SVector(fρ, fρw / ρ_i, fρe)
end

function tendency!(
    ∂q,
    q,
    t,
    H,
    mesh,
    elem,
    grav,
    p_,
    bc = (nothing, nothing),
    forcing = nothing,
)
    fill!(∂q, 0)

    # Evaluate the volume terms
    element_split_form!(∂q, q, elem, H, mesh.zdg)

    # XXX: Hack in the bcs
    ρ = q[1, 1, 1]
    w = q[1, 1, 2]
    ρe = q[1, 1, 3]
    p = p_(ρ, w, ρe, mesh.zdg[1], grav)
    ∂q[1, 1, 2] -= p / ρ
    ρ = q[end, end, 1]
    w = q[end, end, 2]
    ρe = q[end, end, 3]
    p = p_(ρ, w, ρe, mesh.zdg[end], grav)
    ∂q[end, end, 2] += p / ρ

    # Add in MMS formcing
    if !isnothing(forcing)
        error("not implemented yet")
    end

    # Use upwind boundary treatment
    if !isnothing(bc[1])
        error("not implemented yet")
    end

    if !isnothing(bc[2])
        error("not implemented yet")
    end

    # DG -> CG projection and then scatter for storage
    dg2cg_scatter(∂q, mesh)
end

let
  T = Float64
  K = 10
  N = 5
  nflds = 3
  #
  O = element_operators(N)
  M = create_mesh(K, O, T(0), T(30e4))
  #
  Nq = N + 1
  #
  params = Params{T}()
  #
  q = rand(Nq, K, nflds)
  q[:,:,3] .+= 100
  p = similar(M.zdg)
  ρ = @view q[:, :, 1]
  for (i, z) in enumerate(M.zdg)
    (p[i], ρ[i]) = decaying_temperature_profile(params, z)
  end
  q[:, :, 2] .= 0
  ρe = @view q[:, :, 3]
  ρe .= p / (params.γ - 1) + params.grav * ρ .* M.zdg
  #
  γ = T(7 // 5)
  p_(x...) = pressure(x..., γ)
  #
  ∂q = similar(q)
  H(x...) = KennedyGruber(x..., params.grav, p_)
  rhs!(∂q, q, t) = tendency!(∂q, q, t, H, M, O, params.grav, p_)
  c = maximum(wave_speed.(q[:,:,1], p_.(q[:,:,1], q[:,:,2], q[:,:,3], M.zdg, params.grav), γ))
  dt = M.Δz / (N^2 * c)
  @show extrema(q[:, :, 1])
  @show extrema(q[:, :, 2])
  @show extrema(q[:, :, 3])
  day_sec = 86400.0
  for day in 1:100
    tspan = ((day-1) * day_sec, day * day_sec)
    timestep!(q, rhs!, dt, tspan)
    println()
    @show day
    @show extrema(q[:, :, 1])
    @show extrema(q[:, :, 2])
    @show extrema(q[:, :, 3])
  end
  nothing
end
