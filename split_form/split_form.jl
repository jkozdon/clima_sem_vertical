include("common.jl")

function balanced_ρ(p, grav, mesh, elm)
  zdg = mesh.zdg
  Nq, K = size(zdg)
  𝟙 = ones(Bool, Nq)

  pdg = reshape(length(p) == Nq * K ? p : mesh.scatter * p, Nq, K)
  Q = elm.Q

  # Form the RHS vector
  pdg_diff = similar(zdg)
  Fρdg = similar(zdg, Nq * K, Nq * K)
  for e in 1:K
    p̄ = (pdg[:, e] .+ pdg[:, e]') / 2
    pdg_diff[:, e] = ((Q - Q') .* p̄) * 𝟙
  end
  # BCs hack
  pdg_diff[1,1] -= pdg[1,1]
  pdg_diff[end,end] += pdg[end,end]
  p_diff = mesh.scatter' * pdg_diff[:]

  # Form the ρsystem
  T = eltype(zdg)
  Fρdg = zeros(T, Nq * K, Nq * K)
  for e in 1:K
    rng = (e - 1) * Nq .+ (1: Nq)
    Δϕ = grav * (zdg[:, e]' .- zdg[:, e]) / 2
    A = (Q - Q') .* Δϕ
    Fρdg[rng, rng] .= ((Diagonal(A * 𝟙) + A) / 2 )
  end
  Fρ = mesh.scatter' * Fρdg * mesh.scatter
  ρ_bal = -Fρ \ p_diff
  return ρ_bal
end

let
  T = Float64
  K = 10
  N = 5
  nflds = 3
  #
  elem = element_operators(N)
  mesh = create_mesh(K, elem, T(0), T(30e4))
  #
  Nq = N + 1
  #
  params = Params{T}()
  #
  q = rand(Nq, K, nflds)
  q[:,:,3] .+= 100
  p = similar(mesh.zdg)
  ρ = @view q[:, :, 1]
  for (i, z) in enumerate(mesh.zdg)
    (p[i], ρ[i]) = decaying_temperature_profile(params, z)
  end
  ρ[:] .= mesh.scatter * balanced_ρ(p, params.grav, mesh, elem)
  q[:, :, 2] .= 0
  ρe = @view q[:, :, 3]
  ρe .= p / (params.γ - 1) + params.grav * ρ .* mesh.zdg
  #
  γ = T(7 // 5)
  p_(x...) = pressure(x..., γ)
  #
  ∂q = similar(q)
  H(x...) = KennedyGruber(x..., params.grav, p_)
  rhs!(∂q, q, t) = tendency!(∂q, q, t, H, mesh, elem, params.grav, p_)
  c = maximum(wave_speed.(q[:,:,1], p_.(q[:,:,1], q[:,:,2], q[:,:,3], mesh.zdg, params.grav), γ))
  dt = mesh.Δz / (N^2 * c)
  @show extrema(q[:, :, 1])
  @show extrema(q[:, :, 2])
  @show extrema(q[:, :, 3])
  day_sec = 86400.0
  for day in 10:10:100
    tspan = ((day-10) * day_sec, day * day_sec)
    timestep!(q, rhs!, dt, tspan)
    println()
    @show day
    @show extrema(q[:, :, 1])
    @show extrema(q[:, :, 2])
    @show extrema(q[:, :, 3])
  end
  nothing
end
