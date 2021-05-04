include("cg_sem.jl")
using ForwardDiff: derivative
using LinearAlgebra: eigen

function mms_test(N, K, z0, z1, cfl)
    elem = element_operators(N)
    mesh = create_mesh(K, elem, z0, z1)

    problem = Isothermal{Float64}()

    # mms solution
    f(z, t) = cos(t) * cos(2π * (z - z0) / (z1 - z0))
    ρ′(z, t) = f(1z, 4t)
    w′(z, t) = f(2z, 2t)
    E′(z, t) = f(4z, 1t)

    ∂tρ′(z, t) = derivative(t -> ρ′(z, t), t)
    ∂tw′(z, t) = derivative(t -> w′(z, t), t)
    ∂tE′(z, t) = derivative(t -> E′(z, t), t)

    ∂zρ̄w(z, t) = derivative(z -> ρ̄_(problem, z) * w′(z, t), z)
    ∂zδp(z, t) = derivative(z -> δp_(problem, z, E′(z, t), ρ′(z, t)), z)
    ∂zĒ_p̄w′(z, t) =
        derivative(z -> (Ē_(problem, z) + p̄_(problem, z)) * w′(z, t), z)

    # Forcing for each term
    fρ(z, t) = ∂tρ′(z, t) + ∂zρ̄w(z, t)

    grav_source(z, t) = problem.g * ρ′(z, t)
    fw(z, t) = ∂tw′(z, t) + (grav_source(z, t) + ∂zδp(z, t)) / ρ̄_(problem, z)

    fE(z, t) = ∂tE′(z, t) + ∂zĒ_p̄w′(z, t)

    forcing = (ρ = fρ, w = fw, E = fE)

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
        (A = A0, A⁺ = A0⁺, A⁻ = A0⁻, δw = t -> w′(z0, t)),
        (A = A1, A⁺ = A1⁺, A⁻ = A1⁻, δw = t -> w′(z1, t)),
    )

    # Calculate the wave speed
    c̄ = maximum(c̄_(problem, mesh.zdg))

    # estimate time step
    dt = cfl * mesh.Δz / (N^2 * c̄)

    # Make sure we end at the right time
    tspan = (0, 11π)
    steps = ceil(Int, (tspan[2] - tspan[1]) / dt)
    dt = (tspan[2] - tspan[1]) / steps

    # Initial conditions
    q = (
        δρ = ρ′.(mesh.zdg, tspan[1]),
        δw = w′.(mesh.zdg, tspan[1]),
        δE = E′.(mesh.zdg, tspan[1]),
    )

    # advance in time
    rhs!(∂q, q, t) = tendency!(∂q, q, t, problem, mesh, elem, bc, forcing)
    timestep!(q, rhs!, dt, tspan)

    # Exact solution at the final time
    qe = (
        δρ = ρ′.(mesh.zdg, tspan[2]),
        δw = w′.(mesh.zdg, tspan[2]),
        δE = E′.(mesh.zdg, tspan[2]),
    )

    # Calculate the L2 error
    ϵ = zeros(3)
    for i in 1:3
        ϵ[i] = sqrt(sum(mesh.J .* elem.ω .* (q[i] - qe[i]) .^ 2))
    end

    return ϵ
end

function mms_run(N, Ks; cfl = 1)
    ϵ = zeros(3, length(Ks))
    for (lvl, K) in enumerate(Ks)
        @show lvl, K
        ϵ[:, lvl] .= mms_test(N, K, 0, 100π, cfl)
        @show ϵ[:, lvl]
        if lvl > 1
            rate =
                (log.(ϵ[:, lvl]) - log.(ϵ[:, lvl - 1])) ./
                (log.(Ks[lvl - 1]) - log.(Ks[lvl]))
            @show rate
        end
        println()
    end
end
