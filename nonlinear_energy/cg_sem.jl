include(joinpath("..", "common", "utils.jl"))
using LinearAlgebra: Diagonal, diag

⊗ = kron

function element_operators(N, T = Float64)
    ξ, ω = lglpoints(BigFloat, N)
    D = spectralderivative(ξ)
    return (ξ = T.(ξ), ω = T.(ω), D = T.(D))
end

wave_speed(ρ, p, γ) = sqrt(γ * p ./ ρ)

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

function dg2cg_scatter(q, mesh, elem)
    Q = mesh.scatter
    ω = elem.ω
    if q isa Array
        q[:] .= Q * ((Q' * (mesh.J * ω .* q)[:]) ./ mesh.Wcg)
    elseif q isa Tuple || q isa NamedTuple
        for i in 1:length(q)
            q[i][:] .= Q * ((Q' * (mesh.J * ω .* q[i])[:]) ./ mesh.Wcg)
        end
    end
    return q
end

function legendre_polynomial(ξ, P)
    T = eltype(ξ)
    N = length(ξ) - 1
    coef = GaussQuadrature.legendre_coefs(T, max(1, P))
    v = GaussQuadrature.orthonormal_poly(ξ, coef...)[:, P + 1]
    return v
end

function balanced_rho(p, grav, mesh, elem)
    Nq = length(elem.ξ)
    N = Nq - 1

    # Compute the derivative
    ρ = -elem.D * p / (mesh.J * grav)
    ρ0 = copy(ρ)

    # make continuous by adding a higher mode on each element
    v = legendre_polynomial(elem.ξ, N)
    v ./= v[1]
    for e in 2:size(ρ, 2)
        Δρ = ρ[1, e] - ρ[Nq, e - 1]
        ρ[:, e] .-= Δρ * v
    end

    return ρ
end

"""
    element_tendency!(∂q, q, elem_operator, problem, z_elem)

Evaluate the element rhs `∂q` associated with `q` for the
`problem::AbstractProblem` using the DG `elem_operator`.
`q` and `∂q` are taken to be `NamedTuple`s with fields `(ρ, w, E)`
"""
function element_tendency!(∂q, q, O, J, z, grav, p_, Pr = I)
    ρ, w, ρe = q.ρ, q.w, q.ρe
    p = p_.(ρ, w, ρe, z)

    # ∫ ϕ J ∂ρ = ∫ (∂_ξ ϕ) ρ w
    ∂q.ρ .= (O.D' * (O.ω .* ρ .* w)) ./ (J .* O.ω)
    # ∂q.ρ .= (ρ .* (O.D' * (O.ω .* w))) ./ (J .* O.ω)
    # ∂q.ρ .-= w .* (O.D * ρ) ./ J

    #  ∫ ϕ ∂w = - ∫ ϕ (g - (∂_ξ p) / ρ̂) + ∫ (∂_ξ ϕ) w^2/2
    #  where ρ̂ has the projection operator Pr applied
    ∂q.w .= -grav .- (O.D * p) ./ (J .* Pr * ρ)

    #  ∫ ϕ ∂w = - ∫ ϕ ((g ρ̂  - ∂_ξ p) / ρ) + ∫ (∂_ξ ϕ) w^2/2
    #  where ρ̂ has the projection operator Pr applied
    # ∂q.w .= -(grav * (Pr * ρ) + (O.D * p) / J) ./ ρ
    ∂q.w .+= (O.D' * (O.ω .* w .^ 2 / 2)) ./ (J .* O.ω)

    #  ∫ ϕ ∂ρe = ∫ (∂_ξ ϕ) (ρe + p) w
    ∂q.ρe .= (O.D' * (O.ω .* (ρe + p) .* w)) ./ (J .* O.ω)

    return ∂q
end

function tendency!(
    ∂q,
    q,
    t,
    mesh,
    elem,
    grav,
    pres,
    Pr = I,
    bc = (nothing, nothing),
    forcing = nothing,
)
    # Evaluate the volume terms
    element_tendency!(∂q, q, elem, mesh.J, mesh.zdg, grav, pres, Pr)

    # Add in MMS formcing
    if !isnothing(forcing)
        ∂q.δρ .+= forcing.ρ.(mesh.zdg, t)
        ∂q.δw .+= forcing.w.(mesh.zdg, t)
        ∂q.δE .+= forcing.ρe.(mesh.zdg, t)
    end

    # Use upwind boundary treatment
    if !isnothing(bc[1])
        error("not implemented yet")
    end

    if !isnothing(bc[2])
        error("not implemented yet")
    end

    # DG -> CG projection and then scatter for storage
    dg2cg_scatter(∂q, mesh, elem)
end
