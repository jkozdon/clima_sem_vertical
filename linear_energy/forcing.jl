include("cg_sem.jl")
using LinearAlgebra: eigen

function boundary_forcing_test(
    N,
    K,
    z0,
    z1;
    δw0 = t -> 0,
    δw1 = t -> 0,
    cfl = 1,
    tspan = (0, 1),
)
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
        (A = A0, A⁺ = A0⁺, A⁻ = A0⁻, δw = δw0),
        (A = A1, A⁺ = A1⁺, A⁻ = A1⁻, δw = δw1),
    )

    # Calculate the wave speed
    c̄ = maximum(c̄_(problem, mesh.zdg))

    # estimate time step
    dt0 = cfl * mesh.Δz / (N^2 * c̄)

    # Initial conditions
    q = (
        δρ = fill!(similar(mesh.zdg), 0),
        δw = fill!(similar(mesh.zdg), 0),
        δE = fill!(similar(mesh.zdg), 0),
    )

    # advance in time
    rhs!(∂q, q, t) = tendency!(∂q, q, t, problem, mesh, elem, bc)

    for step in 1:length(tspan)
        if step > 1
            ts = (tspan[step - 1], tspan[step])
            steps = ceil(Int, (ts[2] - ts[1]) / dt0)
            dt = (ts[2] - ts[1]) / steps
            timestep!(q, rhs!, dt, ts)
        end

        # Calculate the L2 error
        L2 = zeros(3)
        for i in 1:3
            L2[i] = sqrt(sum(mesh.J .* elem.ω .* q[i] .^ 2))
        end
        t = tspan[step]
        @show t, L2
    end
end
