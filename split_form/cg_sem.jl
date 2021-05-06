include(joinpath("..", "common", "utils.jl"))

using LinearAlgebra: Diagonal, diag

⊗ = kron

function element_operators(N, T = Float64)
    ξ, ω = lglpoints(BigFloat, N)
    D = spectralderivative(ξ)
    Q = Diagonal(ω) * D
    return (ξ = T.(ξ), ω = T.(ω), D = T.(D), Q = T.(Q))
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

function dg2cg_scatter(q, mesh)
    Q = mesh.scatter
    for f in 1:size(q, 3)
      qf = @view q[:, :, f]
      qf[:] .= Q * ((Q' * qf[:]) ./ mesh.Wcg)
    end
    return q
end
