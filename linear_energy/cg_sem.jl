include(joinpath("..", "common", "utils.jl"))
using SparseArrays: spzeros
using Logging: @info
using Printf: @sprintf
using LinearAlgebra: diag, eigen
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

p̄(s::AbstractProblem, z) = s.p₀ * exp(-z / s.H)
∂zp̄(s::AbstractProblem, z) = -s.H * s.p₀ * exp(-z / s.H)

ρ̄(s::AbstractProblem, z) = s.ρ₀ * exp(-z / s.H)
∂zρ̄(s::AbstractProblem, z) = -s.H * s.ρ₀ * exp(-z / s.H)

Ē(s::AbstractProblem, z) = ρ̄(s, z) * (s.Cᵥ * s.T₀ + s.g * z)
∂zĒ(s::AbstractProblem, z) = ∂zρ̄(s, z) * (s.Cᵥ * s.T₀ + s.g * z) + ρ̄(s, z) * s.g

δp_(s::AbstractProblem, z, δE, δρ) = (s.R / s.Cᵥ) * (δE - s.g * z * δρ)
∂zδp(s::AbstractProblem, z, δE, δρ, ∂zδE, ∂zδρ) = (s.R / s.Cᵥ) * (∂zδE - s.g * z * ∂zδρ - s.g * δρ)

A_(s, z) = [0 1 0
           -(s.R / s.Cᵥ) * s.g * z 0 (s.R / s.Cᵥ)
           0 (Ē(s, z) + p̄(s, z)) / ρ̄(s, z) 0]

function build_operator(s::AbstractProblem, N, K, (z0, z1)::NTuple{2, T}) where T
  # for tensor product construction
  I_KK = sparse(I, K, K)

  # Create the element operators
  ξ, ω = lglpoints(T, N)

  # Get the scatter matrix
  Q = scatter_matrix(N, K)

  # cell size
  Δz = (z1 - z0) / K

  z = [z0 .+ Δz * ((ξ[1:end-1] .+ 1) / 2 .+ (0:(K-1))')[:]; z1]

  zdg = Q * z

  # Jacobian determinant
  J = Δz / 2

  # Form the grid mass matrices
  W = J * I_KK ⊗ Diagonal(ω)

  _ρ̄ = ρ̄.(Ref(s), zdg)
  _p̄ = p̄.(Ref(s), zdg)
  _Ē = Ē.(Ref(s), zdg)

  # CG mass matrices
  M = Q' * W * Q
  Mρ̄ = Q' * W * Diagonal(_ρ̄) * Q

  MI = Diagonal(1 ./ diag(M))
  Mρ̄I = Diagonal(1 ./ diag(Mρ̄))

  # CG stiffness matrix
  D = I_KK ⊗ spectralderivative(ξ) / J

  S = Q' * W * D * Q
  Sρ̄ = Q' * Diagonal(_ρ̄) * W * D * Q
  SĒp̄ = Q' * Diagonal(_Ē + _p̄) * W * D * Q

  Np = N * K - 1

  (z = z, MI = MI, Mρ̄I = Mρ̄I, S = S, Sρ̄ = Sρ̄, SĒp̄ = SĒp̄, M = M)
end

function tendency!(s, O, (∂δρ, ∂δw, ∂δE), (δρ, δw, δE), t,
    (fρ, fw, fE) = (nothing, nothing, nothing), 
    (A0, A1) = (nothing, nothing),
    (δw0, δw1) = (0, 0))

  z = O.z

  δp = δp_.(Ref(s), z, δE, δρ)

  ∂δρ .+= O.MI * O.Sρ̄' * δw
  ∂δw .+= -O.Mρ̄I * (O.S * δp + s.g * O.M * δρ)
  ∂δE .+= O.MI * O.SĒp̄' * δw

  if !isnothing(A0)
    ρ̄0 = ρ̄(s, z[1])
    x = [δρ[1], ρ̄0 * δw[1], δE[1]]
    y = [x[1], -x[2] + 2ρ̄0 * δw0, x[3]]
    f0 = A0.out * x + A0.in * y
    ∂δρ[1] += O.MI[1,1] * f0[1]
    flw = -(s.R / s.Cᵥ) * s.g * z[1] * x[1] + (s.R / s.Cᵥ) * x[2]
    ∂δw[1] += O.Mρ̄I[1,1] * (f0[2] - flw)
    ∂δE[1] += O.MI[1,1] * f0[3]
  end
  if !isnothing(A1)
    ρ̄1 = ρ̄(s, z[end])
    x = [δρ[end], ρ̄1 * δw[end], δE[end]]
    y = [x[1], -x[2] + 2ρ̄1 * δw1, x[3]]
    f1 = A1.out * x + A1.in * y
    ∂δρ[end] -= O.MI[ end,end] * f1[1]
    flw = -(s.R / s.Cᵥ) * s.g * z[end] * x[1] + (s.R / s.Cᵥ) * x[2]
    ∂δw[end] -= O.Mρ̄I[end,end] * (f1[2] - flw)
    ∂δE[end] -= O.MI[ end,end] * f1[3]
  end

  # If we need to add an MMS forcing
  isnothing(fρ) || (∂δρ .+= fρ.(z, t))
  isnothing(fw) || (∂δw .+= fw.(z, t))
  isnothing(fE) || (∂δE .+= fE.(z, t))

  return (∂δρ, ∂δw, ∂δE)
end

let
  Ks = 2 .^ (2:7)
  error = ntuple(i->zeros(length(Ks)), 3)

  s = Isothermal{Float64}()
  (z0, z1) = (0.0, 30000.0)
  N = 3

  f(z, t) = cos(t) * cos(2π * (z - z0) / (z1 - z0))
  ρ′, w′, E′ = f, f, f

  fρ(z, t) = derivative(t->ρ′(z, t), t) + derivative(z->ρ̄(s, z) * w′(z, t), z)
  fw(z, t) = derivative(t->ρ′(z, t), t) +
  (s.g * ρ′(z, t) + derivative(z->δp_(s, z, E′(z, t), ρ′(z, t)), z)) / ρ̄(s, z)
  fE(z, t) = derivative(t->E′(z, t), t) + derivative(z->(Ē(s, z) + p̄(s, z)) * w′(z, t), z)

  A = A_(s, z0)
  (λ, V) = eigen(A)
  A⁺ = V * Diagonal(max.(0, λ)) / V
  A⁻ = V * Diagonal(min.(0, λ)) / V
  A0 = (in = A⁺, out = A⁻)

  A = A_(s, z1)
  (λ, V) = eigen(A)
  A⁺ = V * Diagonal(max.(0, λ)) / V
  A⁻ = V * Diagonal(min.(0, λ)) / V
  A1 = (in = A⁻, out = A⁺)

  for (lvl, K) = enumerate(Ks)
    O = build_operator(s, N, K, (z0, z1))

    rhs!(∂q, q, t) = tendency!(s, O, ∂q, q, t, (fρ, fw, fE), (A0, A1),
                               (w′(z0, t), w′(z1, t)))

    steps = 100K
    tspan = (0, 11π)
    dt = tspan[2] / steps
    δq = ntuple(i-> f.(O.z, tspan[1]), 3)
    timestep!(δq, rhs!, dt, tspan)
    δq_f = ntuple(i-> f.(O.z, tspan[2]), 3)
    for i = 1:3
      error[i][lvl] = sqrt((δq[i] - δq_f[i])' * O.M * (δq[i] - δq_f[i]))
    end
    @show (error[1][lvl], error[2][lvl], error[3][lvl])
  end

  for i = 1:3
    rate = (log.(error[i][1:end-1]) - log.(error[i][2:end])) ./
    (log.(Ks[2:end]) - log.(Ks[1:end-1]))
    @show rate
  end
end

#=
begin
  K = 100
  tspan = (0, 300_000)

  s = Isothermal{Float64}()
  (z0, z1) = (0.0, 30_000_000.0)
  N = 3

  A = A_(s, z0)
  (λ, V) = eigen(A)
  A⁺ = V * Diagonal(max.(0, λ)) / V
  A⁻ = V * Diagonal(min.(0, λ)) / V
  A0 = (in = A⁺, out = A⁻)

  A = A_(s, z1)
  (λ, V) = eigen(A)
  A⁺ = V * Diagonal(max.(0, λ)) / V
  A⁻ = V * Diagonal(min.(0, λ)) / V
  A1 = (in = A⁻, out = A⁺)

  O = build_operator(s, N, K, (z0, z1))

  rhs!(∂q, q, t) = tendency!(s, O, ∂q, q, t,
                             (nothing, nothing, nothing),
                             (A0, A1), (1, 0))
  c̄ = 300
  dt = z1 / K / 400 / (N + 1) / 100

  steps = ceil(Int, tspan[2] / dt)
  dt = tspan[2] / steps
  δq = ntuple(i-> fill!(similar(O.z), 0), 3)
  timestep!(δq, rhs!, dt, tspan)
  δq
end
=#
