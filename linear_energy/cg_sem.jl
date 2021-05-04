include(joinpath("..", "common", "utils.jl"))
using LinearAlgebra: Diagonal, diag

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
    γ::T
    function Isothermal{T}(;
        T₀ = 300,
        R = 287,
        p₀ = 10^5,
        g = T(98 // 100),
    ) where {T}
        H = R * T₀ / g
        ρ₀ = p₀ / (R * T₀)
        Cᵥ = 5R / 2
        γ = R / Cᵥ + 1
        new{T}(T₀, R, p₀, g, ρ₀, H, Cᵥ, γ)
    end
end

p̄_(s::AbstractProblem, z) = s.p₀ * exp(-z / s.H)
ρ̄_(s::AbstractProblem, z) = s.ρ₀ * exp(-z / s.H)
Ē_(s::AbstractProblem, z) = ρ̄_(s, z) * (s.Cᵥ * s.T₀ + s.g * z)
δp_(s::AbstractProblem, z, δE, δρ) = (s.R / s.Cᵥ) * (δE - s.g * z * δρ)
c̄_(s::AbstractProblem, _) = sqrt(s.γ * s.p₀ / s.ρ₀)

A_(s, z) = [
    0 1 0
    -(s.R / s.Cᵥ)*s.g*z 0 (s.R/s.Cᵥ)
    0 (Ē_(s, z) + p̄_(s, z))/ρ̄_(s, z) 0
]

function element_operators(N, T = Float64)
    ξ, ω = lglpoints(T, N)
    D = spectralderivative(ξ)
    return (ξ = ξ, ω = ω, D = D)
end

function create_mesh(K, elem, z0, z1)
    ξ = elem.ξ

    # Number of pointss in element and polynomial order
    Nq = length(ξ)
    N = Nq - 1

    # Element size
    Δz = (z1 - z0) / K
    J = Δz / (ξ[Nq] - ξ[1])

    # shift ξ to go (0, 1)
    ξ01 = (ξ[1:(end - 1)] .- ξ[1]) / (ξ[Nq] - ξ[1])

    # cg to dg scatter matrix
    Q = scatter_matrix(N, K)

    # CG DOF locations
    zcg = [z0 .+ Δz * (ξ01 .+ (0:(K - 1))')[:]; z1]

    # DG DOF locations
    zdg = reshape(Q * zcg, Nq, K)

    W = Diagonal(elem.ω)
    I_KK = sparse(I, K, K)
    Wcg = Array(diag(Q' * (I_KK ⊗ (J * W)) * Q))

    return (zcg = zcg, zdg = zdg, scatter = Q, J = J, Δz = Δz, Wcg = Wcg)
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
    ∂q.δw .= -s.g * q.δρ ./ ρ̄ - (O.D * δp) ./ (J .* ρ̄)

    #  ∫ ϕ ∂δE = ∫ (∂_ξ ϕ) (Ē + p̄) δw
    ∂q.δE .= (O.D' * (O.ω .* (Ē + p̄) .* q.δw)) ./ (J .* O.ω)
end

function dg2cg_scatter(q, mesh, elem)
    Q = mesh.scatter
    ω = elem.ω
    for i in 1:length(q)
        q[i][:] .= Q * ((Q' * (mesh.J * ω .* q[i])[:]) ./ mesh.Wcg)
    end
end

function tendency!(∂q, q, t, problem, mesh, elem, bc, forcing = nothing)
    # Evaluate the volume terms
    element_tendency!(∂q, q, elem, mesh.J, problem, mesh.zdg)

    # Add in MMS formcing
    if !isnothing(forcing)
        ∂q.δρ .+= forcing.ρ.(mesh.zdg, t)
        ∂q.δw .+= forcing.w.(mesh.zdg, t)
        ∂q.δE .+= forcing.E.(mesh.zdg, t)
    end

    # Use upwind boundary treatment
    if !isnothing(bc[1])
        # Get the boundary velocity and reference density
        δw0 = bc[1].δw(t)
        ρ̄0 = ρ̄_(problem, mesh.zdg[1])

        # Set the plus and minus states
        q⁻ = [q.δρ[1], ρ̄0 * q.δw[1], q.δE[1]]
        q⁺ = [q⁻[1], -q⁻[2] + 2ρ̄0 * δw0, q⁻[3]]

        # Evaluate the upwind flux
        f0 = bc[1].A⁻ * q⁻ + bc[1].A⁺ * q⁺

        # Evaluate the minus-side flux for w (strong derivative)
        flw = bc[1].A[2, :]' * q⁻

        # Apply lift
        ∂q.δρ[1] += f0[1] / (mesh.J * elem.ω[1])
        ∂q.δw[1] += (f0[2] - flw) / (mesh.J * elem.ω[1] * ρ̄0)
        ∂q.δE[1] += f0[3] / (mesh.J * elem.ω[1])
    end

    if !isnothing(bc[2])
        # Get the boundary velocity and reference density
        δw1 = bc[2].δw(t)
        ρ̄1 = ρ̄_(problem, mesh.zdg[end])

        # Set the plus and minus states
        q⁻ = [q.δρ[end], ρ̄1 * q.δw[end], q.δE[end]]
        q⁺ = [q⁻[1], -q⁻[2] + 2ρ̄1 * δw1, q⁻[3]]

        # Evaluate the upwind flux
        f1 = bc[2].A⁻ * q⁻ + bc[2].A⁺ * q⁺

        # Evaluate the minus-side flux for w (strong derivative)
        flw = bc[2].A[2, :]' * q⁻

        # Apply lift
        ∂q.δρ[end] += f1[1] / (mesh.J * elem.ω[end])
        ∂q.δw[end] += (f1[2] - flw) / (mesh.J * elem.ω[end] * ρ̄1)
        ∂q.δE[end] += f1[3] / (mesh.J * elem.ω[end])
    end

    # DG -> CG projection and then scatter for storage
    dg2cg_scatter(∂q, mesh, elem)
end
