include("common.jl")
using LinearAlgebra: rank, eigvals

function balanced_ρ(p, grav, mesh, elm)
  zdg = mesh.zdg
  Nq, K = size(zdg)
  𝟙 = ones(Bool, Nq)

  pdg = reshape(mesh.scatter * p, Nq, K)
  Q = elm.Q

  # Form the RHS vector
  pdg_diff = similar(zdg)
  Fρdg = similar(zdg, Nq * K, Nq * K)
  for e in 1:K
    p̄ = (pdg[:, e] .+ pdg[:, e]') / 2
    pdg_diff[:, e] = ((Q - Q') .* p̄) * 𝟙
  end
  # BCs hack
  pdg_diff[1,1] -= p[1,1]
  pdg_diff[end,end] += p[end,end]
  p_diff = mesh.scatter' * pdg_diff[:]
  @show p_diff

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
  #
  elm = element_operators(N)
  mesh = create_mesh(K, elm, T(0), T(30e4))
  zcg = mesh.zcg
  #
  Nq = N + 1
  #
  params = Params{T}()
  p = similar(zcg)
  ρ = similar(zcg)
  for (i, z) in enumerate(zcg)
    (p[i], ρ[i]) = decaying_temperature_profile(params, z)
  end
  #
  ρdg = reshape(mesh.scatter * ρ, Nq, K)
  pdg = reshape(mesh.scatter * p, Nq, K)
  #
  vdg = fill!(similar(ρdg), 0)
  #
  Q = elm.Q
  zdg = mesh.zdg
  grav = params.grav
  #
  pdg_diff_0 = fill!(similar(pdg), 0)
  ρdg_diff_0 = fill!(similar(ρdg), 0)
  for e in 1:K
    for i in 1:Nq, j in 1:Nq
      p_i, p_j = pdg[i, e], pdg[j, e]
      ρ_i, ρ_j = ρdg[i, e], ρdg[j, e]
      ϕ_i, ϕ_j = grav * zdg[i, e], grav * zdg[j, e]
      # flux terms
      ρ̄ = (ρ_i + ρ_j) / 2
      Δϕ = (ϕ_j - ϕ_i) / 2
      p̄ = (p_i + p_j) / 2
      # flux differences
      pdg_diff_0[i, e] += (Q[i, j] - Q[j, i]) * p̄
      ρdg_diff_0[i, e] += (Q[i, j] - Q[j, i]) * ρ̄ * Δϕ
    end
  end
  # BCs hack
  pdg_diff_0[1,1] -= p[1,1]
  pdg_diff_0[end,end] += p[end,end]
  #
  # @show extrema(mesh.scatter'*(pdg_diff_0 + ρdg_diff_0)[:])
  #
  # Remove the inner for loops
  pdg_diff_1 = fill!(similar(pdg), 0)
  ρdg_diff_1 = fill!(similar(ρdg), 0)
  𝟙 = ones(Bool, Nq)
  for e in 1:K
    p̄ = (pdg[:, e] .+ pdg[:, e]') / 2
    ρ̄ = (ρdg[:, e] .+ ρdg[:, e]') / 2
    Δϕ = grav * (zdg[:, e]' .- zdg[:, e]) / 2
    pdg_diff_1[:, e] = ((Q - Q') .* p̄) * 𝟙
    ρdg_diff_1[:, e] = ((Q - Q') .* ρ̄ .* Δϕ) * 𝟙
  end
  # BCs hack
  pdg_diff_1[1,1] -= p[1,1]
  pdg_diff_1[end,end] += p[end,end]
  @assert all(pdg_diff_0 .≈ pdg_diff_1)
  @assert all(ρdg_diff_0 .≈ ρdg_diff_1)
  #
  # Build the ρ matrix
  Fρdg = zeros(T, Nq * K, Nq * K)
  for e in 1:K
    rng = (e - 1) * Nq .+ (1: Nq)
    Δϕ = grav * (zdg[:, e]' .- zdg[:, e]) / 2
    A = (Q - Q') .* Δϕ
    Fρdg[rng, rng] .= ((Diagonal(A * 𝟙) + A) / 2 )
  end
  ρdg_diff_2 = reshape(Fρdg * ρdg[:], Nq, K)
  @assert all(ρdg_diff_0 .≈ ρdg_diff_2)
  #
  Fρ = mesh.scatter' * Fρdg * mesh.scatter
  ρ_bal = -Fρ \ (mesh.scatter' * pdg_diff_0[:])
  ρdg_bal = reshape(mesh.scatter * ρ_bal, Nq, K)
  display(ρdg)
  display(ρdg_bal)
  display(ρdg_bal - ρdg)
  ρ_bal2 = balanced_ρ(p, grav, mesh, elm)
  @show extrema(ρ_bal2 - ρ_bal)
  nothing
end
