include(joinpath("..", "common", "utils.jl"))
include("cg_sem.jl")
import GaussQuadrature
using PGFPlotsX: @pgf, Plot, Axis, pgfsave, Table, LegendEntry, GroupPlot

Base.@kwdef struct Params{T}
    T_virt_surf::T = 280
    T_min_ref::T = 230
    MSLP::T = 10^5
    grav::T = 98 // 100
    R_d::T = 287
    γ::T = 7 // 5
end

# Taken from Daniel's code
function decaying_temperature_profile(params::Params, z)
    # Scale height for surface temperature
    H_sfc = params.R_d * params.T_virt_surf / params.grav
    H_t = H_sfc

    z′ = z / H_t
    tanh_z′ = tanh(z′)

    ΔTv = params.T_virt_surf - params.T_min_ref
    Tv = params.T_virt_surf - ΔTv * tanh_z′

    ΔTv′ = ΔTv / params.T_virt_surf
    p = -H_t * (z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
    p /= H_sfc * (1 - ΔTv′^2)
    p = params.MSLP * exp(p)
    ρ = p / (params.R_d * Tv)
    return (p = p, ρ = ρ)
end

function plot_comparison(output_file, ρ, ρ_ref, ρe, ρe_ref, z)
    # Plot ρ_ref and ρ
    @pgf ρ_plot = Axis()
    @pgf push!(ρ_plot, Plot({color = "red", no_marks}, Table(ρ_ref[:], z[:])))
    @pgf push!(ρ_plot, LegendEntry(raw"$\rho_{ref}$"))
    @pgf push!(ρ_plot, Plot({color = "blue", no_marks}, Table(ρ[:], z[:])))
    @pgf push!(ρ_plot, LegendEntry(raw"$\rho_{hb}$"))

    # Plot ρ_ref - ρ
    @pgf Δρ_plot = Axis()
    @pgf push!(
        Δρ_plot,
        Plot({color = "red", no_marks}, Table(ρ_ref[:] - ρ[:], z[:])),
    )
    @pgf push!(Δρ_plot, LegendEntry(raw"$\rho_{ref} - \rho_{hb}$"))

    # Plot ρe_ref and ρe
    @pgf ρe_plot = Axis()
    @pgf push!(ρe_plot, Plot({color = "red", no_marks}, Table(ρe_ref[:], z[:])))
    @pgf push!(ρe_plot, LegendEntry(raw"$\rho e_{ref}$"))
    @pgf push!(ρe_plot, Plot({color = "blue", no_marks}, Table(ρe[:], z[:])))
    @pgf push!(ρe_plot, LegendEntry(raw"$\rho e_{hb}$"))

    # Plot ρe_ref - ρe
    @pgf ΔE_plot = Axis()
    @pgf push!(
        ΔE_plot,
        Plot(
            {color = "red", no_marks},
            Table((ρe_ref[:] - ρe[:]) ./ (ρe_ref[:]), z[:]),
        ),
    )
    @pgf push!(
        ΔE_plot,
        LegendEntry(raw"$(\rho e_{ref} - \rho e_{hb})/\rho e_{ref}$"),
    )

    # Make subplots
    @pgf hb_compare = GroupPlot(
        {
            group_style = {
                group_size = "2 by 2",
                horizontal_sep = "50pt",
                vertical_sep = "50pt",
            },
            legend_pos = "north east",
            ymin = minimum(z[:]),
            ymax = maximum(z[:]),
            ylabel = "height",
            # "scaled y ticks = false",
            # "scaled x ticks = false",
        },
        ρ_plot,
        ρe_plot,
        Δρ_plot,
        ΔE_plot,
    )

    pgfsave(output_file, hb_compare)
end

#=
let
    T = Float64
    N = 5
    Nq = N + 1
    K = 10
    #
    z0, z1 = 0, 30e3
    elem = element_operators(N)
    mesh = create_mesh(K, elem, z0, z1)
    #
    p = rand(T, Nq, K)
    #
    dg2cg_scatter(p, mesh, elem)
    #
    params = Params{T}()
    #
    ρ0 = -elem.D * p / (mesh.J * params.grav)
    #
    ρ = balanced_rho(p, params.grav, mesh, elem)
    #
    Pr = DG_cutoff_filter_matrix(elem.ξ, N)
    @assert all(Pr * ρ .≈ ρ0)
    ρ1 = dg2cg_scatter(copy(ρ), mesh, elem)
    @assert all(ρ .≈ ρ1)
end

let
    T = Float64
    N = 4
    Nq = N + 1
    K = 250
    #
    z0, z1 = 0, 30e4
    elem = element_operators(N)
    mesh = create_mesh(K, elem, z0, z1)
    #
    params = Params{T}()
    #
    p, ρref = similar(mesh.zdg), similar(mesh.zdg)
    for (i, z) in enumerate(mesh.zdg)
        (p[i], ρref[i]) = decaying_temperature_profile(params, z)
    end
    #
    ρ_dg = -elem.D * p / (mesh.J * params.grav)
    #
    ρ_cg = balanced_rho(p, params.grav, mesh, elem)
    Pr = DG_cutoff_filter_matrix(elem.ξ, N)
    @assert all(Pr * ρ_cg .≈ ρ_dg)
    #
    nothing
end
=#

let
    T = Float64
    N = 4
    Nq = N + 1
    K = 15
    #
    z0, z1 = 0, 30e4
    elem = element_operators(N)
    mesh = create_mesh(K, elem, z0, z1)
    #
    params = Params{T}()
    #
    p, ρ_ref = similar(mesh.zdg), similar(mesh.zdg)
    for (i, z) in enumerate(mesh.zdg)
        (p[i], ρ_ref[i]) = decaying_temperature_profile(params, z)
    end
    #
    ρ = balanced_rho(p, params.grav, mesh, elem)
    #
    w = fill!(similar(ρ), 0)
    #
    ρe_ref =
        ρ_ref .* w .^ 2 / 2 +
        p / (params.γ - 1) +
        params.grav * ρ_ref .* mesh.zdg
    ρe = ρ .* w .^ 2 / 2 + p / (params.γ - 1) + params.grav * ρ .* mesh.zdg
    #
    #=
    plot_comparison(
        "balance_compare_initial.pdf",
        ρ,
        ρ_ref,
        ρe,
        ρe_ref,
        mesh.zdg,
    )
    #
    ∂q = (
        ρ = fill!(similar(ρ), 0),
        w = fill!(similar(w), 0),
        ρe = fill!(similar(ρe), 0),
    )
    =#
    pres(ρ, w, ρe, z) =
        (params.γ - 1) * (ρe - ρ * w^2 / 2 - ρ * params.grav * z)
    #
    #
    # q = (ρ = ρ_ref, w = w, ρe = ρe_ref)
    q = (ρ = ρ, w = w, ρe = ρe)
    q0 = (ρ = copy(q.ρ), w = copy(q.w), ρe = copy(q.ρe))
    Pr = DG_cutoff_filter_matrix(elem.ξ, N)
    #=
    q_ref = (ρ = ρ_ref, w = w, ρe = ρe_ref)
    #
    element_tendency!(∂q, q_ref, elem, mesh.J, mesh.zdg, params.grav, pres)
    println("Using reference values without projection")
    @show extrema(∂q.w)
    println()
    #
    Pr = DG_cutoff_filter_matrix(elem.ξ, N)
    element_tendency!(∂q, q_ref, elem, mesh.J, mesh.zdg, params.grav, pres, Pr)
    println("Using reference values with projection")
    @show extrema(∂q.w)
    println()
    #
    #
    element_tendency!(∂q, q, elem, mesh.J, mesh.zdg, params.grav, pres)
    println("Using correct values without projection")
    @show extrema(∂q.w)
    println()
    #
    element_tendency!(∂q, q, elem, mesh.J, mesh.zdg, params.grav, pres, Pr)
    println("Using correct values with projection")
    @show extrema(∂q.w)
    println()
    =#
    #
    c = maximum(wave_speed.(ρ, pres.(q.ρ, q.w, q.ρe, mesh.zdg), params.γ))
    #
    dt = mesh.Δz / (N^2 * c)
    #
    @show extrema(q.ρ)
    rhs!(∂q, q, t) = tendency!(∂q, q, t, mesh, elem, params.grav, pres, Pr)
    ndays = 1
    tspan = (0, 86400 * ndays)
    tspan = (0, 100 * dt)
    timestep!(q, rhs!, dt, tspan)
    #
    plot_comparison("balanced.pdf", q0.ρ, q.ρ, q0.ρe, q.ρe, mesh.zdg)
    #
    @show extrema(q0.ρ)
    @show extrema(q.ρ)
    @show extrema(q.ρ - q0.ρ)
    @show extrema(q.w)
    @show sum(elem.ω .* q0.ρ) - sum(elem.ω .* q.ρ)
end
nothing
