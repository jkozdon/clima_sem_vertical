include(joinpath("..", "common", "utils.jl"))
using LinearAlgebra: Diagonal, diag
using ForwardDiff: derivative

⊗ = kron

abstract type AbstractProblem end
struct Isothermal{T} <: AbstractProblem
  T₀::T
  R::T
  p₀::T
  g::T
  ρ₀::T
  H::T
  Cᵥ::T
  function Isothermal{T}(; T₀ = 300, R = 287, p₀ = 10^5, g = T(98 // 100)) where T
    H = R * T₀ / g
    ρ₀ = p₀ / (R * T₀)
    Cᵥ = 5R / 2
    new{T}(T₀, R, p₀, g, ρ₀, H, Cᵥ)
  end
end

p̄_(s::AbstractProblem, z) = s.p₀ * exp(-z / s.H)
ρ̄_(s::AbstractProblem, z) = s.ρ₀ * exp(-z / s.H)
Ē_(s::AbstractProblem, z) = ρ̄_(s, z) * (s.Cᵥ * s.T₀ + s.g * z)
δp_(s::AbstractProblem, z, δE, δρ) = (s.R / s.Cᵥ) * (δE - s.g * z * δρ)

A_(s, z) = [0 1 0
           -(s.R / s.Cᵥ) * s.g * z 0 (s.R / s.Cᵥ)
           0 (Ē(s, z) + p̄(s, z)) / ρ̄(s, z) 0]

function element_operators(N, T = Float64)
  ξ, ω = lglpoints(T, N)
  D = spectralderivative(ξ)
  return (ξ = ξ, ω = ω, D = D)
end

function create_mesh(K, ξ, z0, z1)
  # Number of pointss in element and polynomial order
  Nq = length(ξ)
  N = Nq - 1

  # Element size
  Δz = (z1 - z0) / K

  # shift ξ to go (0, 1)
  ξ01 = (ξ[1:end-1] .- ξ[1]) / (ξ[Nq] - ξ[1])

  # cg to dg scatter matrix
  Q = scatter_matrix(N, K)

  # CG DOF locations
  zcg = [z0 .+ Δz * (ξ01 .+ (0:(K-1))')[:]; z1]

  # DG DOF locations
  zdg = reshape(Q * zcg, Nq, K)

  return (zcg = zcg, zdg = zdg, scatter = Q, J = Δz / (ξ[Nq] - ξ[1]), Δz = Δz)
end

"""
    element_tendency!(∂q, q, elem_operator, problem, z_elem)

Evaluate the element rhs `∂q` associated with `q` for the
`problem::AbstractProblem` using the DG `elem_operator`.
`q` and `∂q` are taken to be `NamedTuple`s with fields `(δρ, δw, δE)`
"""
function element_tendency!(∂q, q, O, J, s, z)
  ρ̄ = ρ̄_.(Ref(s), z)
  Ē = Ē_.(Ref(s), z)
  p̄ = p̄_.(Ref(s), z)

  δp = δp_.(Ref(s), z, q.δE, q.δρ)

  # ∫ ϕ ∂δρ = ∫ (∂_ξ ϕ) ρ̄ δw
  ∂q.δρ .= (O.D' * (O.ω .* ρ̄ .* q.δw)) ./ (J .* O.ω)

  #  ∫ ϕ ∂δw = -∫ (ϕ / ρ̄) (∂_ξ δp + g δρ)
  ∂q.δw .=  - s.g * q.δρ ./ ρ̄ - (O.D * δp) ./ (J .* ρ̄)

  #  ∫ ϕ ∂δE = ∫ (∂_ξ ϕ) (Ē + p̄) δw
  ∂q.δE .= (O.D' * (O.ω .* (Ē + p̄) .* q.δw)) ./ (J .* O.ω)
end

function cg_mass(elem, mesh)
  K = size(mesh.zdg, 2)
  Q = mesh.scatter
  J = mesh.J
  W = Diagonal(elem.ω)
  I_KK = sparse(I, K, K)
  return Array(diag(Q' * (I_KK ⊗ (J * W)) * Q))
end

function dg2cg_scatter(q, mesh, elem, Wcg)
  Q = mesh.scatter
  ω = elem.ω
  for i = 1:length(q)
    q[i][:] .= Q * ((Q' * (mesh.J * ω .* q[i])[:]) ./ Wcg)
  end
end

function main(N, K, z0, z1)
  elem = element_operators(N)
  mesh = create_mesh(K, elem.ξ, z0, z1)
  Wcg = cg_mass(elem, mesh)

  problem = Isothermal{Float64}()

  # mms solution
  f(z, t) = cos(t) * sin(2π * (z - z0) / (z1 - z0))
  ρ′, w′, E′ = f, f, f

  ∂tρ′(z, t) = derivative(t->ρ′(z, t), t)
  ∂tw′(z, t) = derivative(t->w′(z, t), t)
  ∂tE′(z, t) = derivative(t->E′(z, t), t)

  ∂zρ̄w(z, t) = derivative(z->ρ̄_(problem, z) * w′(z, t), z)
  ∂zδp(z, t) = derivative(z->δp_(problem, z, E′(z, t), ρ′(z, t)), z)
  ∂zĒ_p̄w′(z, t) = derivative(z->(Ē_(problem,z) + p̄_(problem,z)) * w′(z, t), z)

  fρ(z, t) = ∂tρ′(z, t) + ∂zρ̄w(z, t)

  grav_source(z, t) = problem.g * ρ′(z, t)
  fw(z, t) = ∂tw′(z, t) + (grav_source(z, t) + ∂zδp(z, t)) / ρ̄_(problem, z)

  fE(z, t) = ∂tE′(z, t) + ∂zĒ_p̄w′(z, t)

  function tendency!(∂q, q, t)
    element_tendency!(∂q, q, elem, mesh.J, problem, mesh.zdg)
    ∂q.δρ .+= fρ.(mesh.zdg, t)
    ∂q.δw .+= fw.(mesh.zdg, t)
    ∂q.δE .+= fE.(mesh.zdg, t)
    dg2cg_scatter(∂q, mesh, elem, Wcg)
  end

  γ = problem.R / problem.Cᵥ + 1
  c̄ = sqrt(maximum(γ * p̄_.(Ref(problem), mesh.zdg) ./ ρ̄_.(Ref(problem), mesh.zdg)))

  dt = mesh.Δz / (N * c̄)
  tspan = (0, 11π)
  steps = ceil(Int, (tspan[2] - tspan[1]) / dt)
  dt = (tspan[2] - tspan[1]) / steps

  q = (δρ = ρ′.(mesh.zdg, tspan[1]),
       δw = w′.(mesh.zdg, tspan[1]),
       δE = E′.(mesh.zdg, tspan[1]),)

  timestep!(q, tendency!, dt, tspan)

  err = zeros(3)
  qe = (δρ = ρ′.(mesh.zdg, tspan[2]),
        δw = w′.(mesh.zdg, tspan[2]),
        δE = E′.(mesh.zdg, tspan[2]),)

  for i = 1:3
    err[i] = sqrt(sum(mesh.J .* elem.ω .* (q[i] - qe[i]).^2))
  end
  @show err
end
