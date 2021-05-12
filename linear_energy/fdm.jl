include("../common/utils.jl")
using ForwardDiff: derivative
using LinearAlgebra: eigen

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

function create_mesh(K, z0, z1)
    Δz = (z1 - z0) / K
    zi = range(z0, stop = z1, length = K + 1)
    zc = zi[1:K] .+ Δz / 2
    return (zc = zc, zi = zi, Δz = Δz)
end

function tendency!(∂q, q, t, s, mesh, f = nothing)
    N = length(q.ρc)

    g = s.g

    Δz = mesh.Δz

    ρ̄i = ρ̄_.(Ref(s), mesh.zi)
    Ēi = Ē_.(Ref(s), mesh.zi)
    p̄i = p̄_.(Ref(s), mesh.zi)

    ρ̄_wi = ρ̄i .* q.wi

    Ē_p̄_wi = (Ēi + p̄i) .* q.wi

    g_ρi = g * (q.ρc[1:(end - 1)] + q.ρc[2:end]) / 2

    pc = δp_.(Ref(s), mesh.zc, q.Ec, q.ρc)

    # ∂zρ̄w
    ∂q.ρc .= -(ρ̄_wi[2:end] - ρ̄_wi[1:(end - 1)]) / Δz

    # (g * ρ′ + ∂zδp) / ρ̄
    ∂q.wi[2:(end - 1)] .= -g_ρi ./ ρ̄i[2:(end - 1)]
    ∂q.wi[2:(end - 1)] .+=
        -(pc[2:end] - pc[1:(end - 1)]) ./ (Δz * ρ̄i[2:(end - 1)])

    # ∂z((Ē + p̄) w′)
    ∂q.Ec .= -(Ē_p̄_wi[2:end] - Ē_p̄_wi[1:(end - 1)]) / Δz

    if !isnothing(f)
      ∂q.ρc .+= f.ρ.(mesh.zc, t)
      ∂q.wi[2:(end - 1)] .+= f.w.(mesh.zi[2:(end - 1)], t)
      ∂q.Ec .+= f.E.(mesh.zc, t)
    end
end

function mms(Ks = 10 * 2 .^ (0:6), z0 = 0.0, z1 = 1.0)
    ϵ = zeros(3, length(Ks))
    for (lvl, K) in enumerate(Ks)
        @show lvl, K

        T = Float64

        mesh = create_mesh(K, z0, z1)

        s = Isothermal{T}()
        # Calculate the wave speed
        c̄ = c̄_(s, 0)

        # mms solution
        κ = c̄ / (z1 - z0)
        f(z, t) = cos(t * κ * π) * sin(2π * (z - z0) / (z1 - z0))
        ρ′(z, t) = f(2z, 4t)
        w′(z, t) = f(1z, 2t)
        E′(z, t) = f(1z, 1t)

        ∂tρ′(z, t) = derivative(t -> ρ′(z, t), t)
        ∂tw′(z, t) = derivative(t -> w′(z, t), t)
        ∂tE′(z, t) = derivative(t -> E′(z, t), t)

        ∂zρ̄w(z, t) = derivative(z -> ρ̄_(s, z) * w′(z, t), z)
        ∂zδp(z, t) = derivative(z -> δp_(s, z, E′(z, t), ρ′(z, t)), z)
        ∂zĒ_p̄w′(z, t) = derivative(z -> (Ē_(s, z) + p̄_(s, z)) * w′(z, t), z)

        # Forcing for each term
        fρ(z, t) = ∂tρ′(z, t) + ∂zρ̄w(z, t)

        grav_source(z, t) = s.g * ρ′(z, t)
        fw(z, t) = ∂tw′(z, t) + (grav_source(z, t) + ∂zδp(z, t)) / ρ̄_(s, z)

        fE(z, t) = ∂tE′(z, t) + ∂zĒ_p̄w′(z, t)

        forcing = (ρ = fρ, w = fw, E = fE)

        cfl = 1

        # estimate time step
        dt = cfl * mesh.Δz / c̄
        tspan = (0, 1 / κ)
        steps = ceil(Int, (tspan[2] - tspan[1]) / dt)
        dt = (tspan[2] - tspan[1]) / steps

        rhs!(∂q, q, t) = tendency!(∂q, q, t, s, mesh, forcing)

        q = (
            ρc = ρ′.(mesh.zc, tspan[1]),
            wi = w′.(mesh.zi, tspan[1]),
            Ec = E′.(mesh.zc, tspan[1]),
        )
        timestep!(q, rhs!, dt, tspan)
        qe = (
            ρc = ρ′.(mesh.zc, tspan[2]),
            wi = w′.(mesh.zi, tspan[2]),
            Ec = E′.(mesh.zc, tspan[2]),
        )

        # Calculate the L2 error
        for i in 1:3
            ϵ[i, lvl] = sqrt(sum(mesh.Δz .* (q[i] - qe[i]) .^ 2))
        end
        @show ϵ[:, lvl]
        if lvl > 1
          rate = (log.(ϵ[:, lvl]) - log.(ϵ[:, lvl-1])) / (log(Ks[lvl-1]) - log(Ks[lvl]))
          @show rate
        end
        println()
    end
end

function form_matrix(K, z0, z1)
    mesh = create_mesh(K, z0, z1)

    s = Isothermal{Float64}()

    # Initial conditions
    q = (
        ρc = fill!(similar(mesh.zc), 0),
        wi = fill!(similar(mesh.zi), 0),
        Ec = fill!(similar(mesh.zc), 0),
    )

    # advance in time
    ∂q = (
        ρc = fill!(similar(mesh.zc), 0),
        wi = fill!(similar(mesh.zi), 0),
        Ec = fill!(similar(mesh.zc), 0),
    )

    rhs!(∂q, q, t) = tendency!(∂q, q, t, s, mesh)

    A = zeros(3K - 1, 3K - 1)
    pts = (ρc = 1:K, wi = K .+ (1:K-1), Ec = 2K-1 .+ (1:K))
    col = 0
    for f = 1:3
      # @show pts[f]
      q[f] .= 0
      ∂q[f] .= 0
    end
    # @show size(A)
    for i = 1:K
      col = col + 1
      q.ρc[i] = 1
      rhs!(∂q, q, 0)
      A[pts.ρc, col] .= ∂q.ρc
      A[pts.wi, col] .= ∂q.wi[2:end-1]
      A[pts.Ec, col] .= ∂q.Ec
      ∂q.ρc .= 0
      ∂q.wi .= 0
      ∂q.Ec .= 0
      q.ρc[i] = 0
    end
    for i = 2:K
      col = col + 1
      q.wi[i] = 1
      rhs!(∂q, q, 0)
      A[pts.ρc, col] .= ∂q.ρc
      A[pts.wi, col] .= ∂q.wi[2:end-1]
      A[pts.Ec, col] .= ∂q.Ec
      ∂q.ρc .= 0
      ∂q.wi .= 0
      ∂q.Ec .= 0
      q.wi[i] = 0
    end
    for i = 1:K
      col = col + 1
      q.Ec[i] = 1
      rhs!(∂q, q, 0)
      A[pts.ρc, col] .= ∂q.ρc
      A[pts.wi, col] .= ∂q.wi[2:end-1]
      A[pts.Ec, col] .= ∂q.Ec
      ∂q.ρc .= 0
      ∂q.wi .= 0
      ∂q.Ec .= 0
      q.Ec[i] = 0
    end

    q = (
        ρc = rand(K),
        wi = rand(K+1),
        Ec = rand(K),
    )
    q.wi[1] = 0
    q.wi[end] = 0

    rhs!(∂q, q, 0)

    q̄ = [q.ρc; q.wi[2:end-1]; q.Ec]
    ∂q̄ = A * q̄

    @assert all(∂q.ρc .≈ ∂q̄[pts.ρc])
    @assert all(∂q.wi[2:end-1] .≈ ∂q̄[pts.wi])
    @assert all(∂q.Ec .≈ ∂q̄[pts.Ec])

    return A
end

let
  K = 1000
  A = form_matrix(K, 0.0, 1.0)
  ev = eigen(A);
  vs = findall(abs.(ev.values) .< 1e-14)
  @show length(vs), K
  pts = (ρc = 1:K, wi = K .+ (1:K-1), Ec = 2K-1 .+ (1:K))
  for v in vs
     @assert maximum(abs.(ev.vectors[pts.wi, v])) < 1e-16
  end
end

