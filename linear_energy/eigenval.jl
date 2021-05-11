include("cg_sem.jl")
using LinearAlgebra: eigen
import Random

function form_matrix(N, K, z0, z1)
    elem = element_operators(N)
    mesh = create_mesh(K, elem, z0, z1)

    problem = Isothermal{Float64}()

    # Matrices for upwinding the boundary conditions
    A0 = A_(problem, z0)
    (λ, V) = eigen(A0)
    A0⁺ = V * Diagonal(max.(0, λ)) / V
    A0⁻ = V * Diagonal(min.(0, λ)) / V

    A1 = -A_(problem, z1)
    (λ, V) = eigen(A1)
    A1⁺ = V * Diagonal(max.(0, λ)) / V
    A1⁻ = V * Diagonal(min.(0, λ)) / V

    bc = (
        (A = A0, A⁺ = A0⁺, A⁻ = A0⁻, δw = t -> 0),
        (A = A1, A⁺ = A1⁺, A⁻ = A1⁻, δw = t -> 0),
    )

    # Initial conditions
    q = (
        δρ = fill!(similar(mesh.zdg), 0),
        δw = fill!(similar(mesh.zdg), 0),
        δE = fill!(similar(mesh.zdg), 0),
    )

    # advance in time
    ∂q = (
        δρ = fill!(similar(mesh.zdg), 0),
        δw = fill!(similar(mesh.zdg), 0),
        δE = fill!(similar(mesh.zdg), 0),
    )
    rhs!(∂q, q, t) = tendency!(∂q, q, t, problem, mesh, elem, bc)

    Npcg = N * K + 1
    v = zeros(Npcg)
    A = zeros(3 * Npcg, 3 * Npcg)
    col = 0
    Nq = N + 1
    cgpts =  [((1:N) .+ Nq * (0:K-1)')[:]..., Nq * K]
    for f = 1:3
      q[f] .= 0
    end
    for f = 1:3
      for i = 1:Npcg
        col = col + 1
        v[i] = 1
        q[f] .= reshape(mesh.scatter * v, N+1, K)
        rhs!(∂q, q, 0)
        for k = 1:3
          A[(k-1) * Npcg .+ (1:Npcg), col] .= ∂q[k][cgpts]
          ∂q[k][cgpts] .= 0
        end
        q[f] .= 0
        v[i] = 0
      end
    end

    # Check the cgpts
    qrnd = rand(Npcg, 3)
    for k = 1:3
      q[k][:] .= mesh.scatter * qrnd[:, k]
      @assert all(q[k][cgpts] .== qrnd[:, k])
    end
    rhs!(∂q, q, 0)
    ∂q2 = reshape(A * qrnd[:], Npcg, 3)
    for k = 1:3
      @assert all(∂q[k] .≈ reshape(mesh.scatter * ∂q2[:, k], Nq, K))
    end

    return A
end

let
  N = 5
  K = 100
  A = A = form_matrix(N, K, 0.0, 30e3)
  ev = eigen(A);
  vs = findall(abs.(ev.values) .< 1e-14)
  @assert length(vs) == N * K + 1
  for v in vs
    @assert all(abs.(reshape(ev.vectors[:, v], N*K+1, 3))[:, 2] .< 1e-15)
  end
  @show extrema(real.(ev.values))
end

