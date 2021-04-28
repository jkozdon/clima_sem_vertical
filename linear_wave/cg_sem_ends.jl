include(joinpath("..", "common", "utils.jl"))
using SparseArrays: spzeros
using Logging: @info
using Printf: @sprintf
using LinearAlgebra: rank, eigvals

⊗ = kron

function build_operator((Nh, Nu), K; T = Float64, α = 1)

  # Polynomial orders of u and h
  @assert Nu ≥ Nh

  I_KK = sparse(I, K, K)

  # Create the element operators
  ξh, ωh = lglpoints(T, Nh)

  # Create the element operators
  ξu, ωu = lglpoints(T, Nu)

  # Get the scatter matrix
  Qh = scatter_matrix(Nh, K)
  Qu = scatter_matrix(Nu, K)

  # domain
  (x0, x1) = (T(0), T(1))

  # cell size
  Δx = (x1 - x0) / K

  xhcg = [x0 .+ Δx * ((ξh[1:end-1] .+ 1) / 2 .+ (0:(K-1))')[:]; x1]
  xucg = [x0 .+ Δx * ((ξu[1:end-1] .+ 1) / 2 .+ (0:(K-1))')[:]; x1]

  # Jacobian determinant
  J = Δx / 2

  # Form the grid mass matrices
  Wh = J * I_KK ⊗ Diagonal(ωh)
  Wu = J * I_KK ⊗ Diagonal(ωu)

  # CG mass matrices
  Mh = Qh' * Wh * Qh
  Mu = Qu' * Wu * Qu

  # CG stiffness matrix
  Ihu = interpolationmatrix(ξh, ξu)
  Dhh = spectralderivative(ξh) / J
  Dhu = Ihu * Dhh
  S = Qu' * Wu * (I_KK ⊗ Dhu) * Qh

  Nph = length(xhcg)
  Npu = length(xucg)

  eu1 = sparse([1], [Npu], [1], 1, Npu)
  eu0 = sparse([1], [1], [1], 1, Npu)

  Sfull = [spzeros(Nph, Nph) -S'
           S -α * eu0' * eu0 - α * eu1' * eu1]

  Mfull = [Mh spzeros(Nph, Npu)
           spzeros(Npu, Nph) Mu]
  Afull = Mfull \ Sfull

  # Return everything as a named tuple
  return (M = Mfull, S = Sfull, A = Afull, xh = xhcg, xu = xucg)
end

function test_operator((Nh, Nu), Ks; final_time = 3, T = Float64)
  @info @sprintf """ 
  (Nh, Nu)   = (%d, %d)
  num levels = %d
  """ Nh Nu length(Ks)

  # Create an exact solution
  # f(x) = exp(sin( T(4) * π * x))
  # g(x) = exp(sin(-T(4) * π * x))
  f(x) = sin( T(4) * π * x)
  g(x) = sin(-T(4) * π * x)

  u_ex(x, t) = (f(x + t) - g(x - t)) / 2
  h_ex(x, t) = (f(x + t) + g(x - t)) / 2

  # Storage for the errors
  ϵ = zeros(T, length(Ks))

  # Loop over the levels
  for (l, K) in enumerate(Ks)
    Op = build_operator((Nh, Nu), K; T = T)

    # build the initial condition
    q0 = [h_ex.(Op.xh, 0); u_ex.(Op.xu, 0)]

    # use exponential time integration because I am lazy
    qf = exp(Matrix(Op.A) * final_time) * q0

    # Calculate the error
    e = qf - [h_ex.(Op.xh, final_time); u_ex.(Op.xu, final_time)]
    ϵ[l] = sqrt(e' * Op.M * e)

    # Display some data
    if l > 1
      rate = ((log(ϵ[l]) - log(ϵ[l-1])) / (log(Ks[l-1]) - log(Ks[l])))
      @info @sprintf """ 
      level = %d
      K     = %d
      error = %.2e
      rate  = %.2e
      """ l K ϵ[l] rate
    else
      @info @sprintf """ 
      level = %d
      K     = %d
      error = %.2e
      """ l K ϵ[l]
    end
  end
end

function mode_check((Nh, Nu), K)
  Op = build_operator((Nh, Nu), K)
  @info @sprintf """ 
  (Nh, Nu)             = (%d, %d)
  number of elements   = %d
  number of zero modes = %d
  """ Nh Nu K size(Op.A, 1) - rank(Op.A)
end

using LinearAlgebra: eigen, norm
using PGFPlotsX
Op = build_operator((5,5), 20; α = 1)
ev = eigen(Matrix(Op.A))

# p = sortperm(abs.(ev.values))
# v1 = ev.vectors[:, p[1]]
vals = ev.values
xmin = min(-1e-10, minimum(real.(vals)))
xmax = max( 1ex-10, maximum(real.(vals)))
@pgf a = Axis(
              {
               xmin = xmin,
               xmax = xmax,
               # ymax = 40,
               ymin = 0,
               xlabel = "\$\\textrm{Re}(\\lambda)\$",
               ylabel = "\$\\textrm{Im}(\\lambda)\$",
              }
             );
@pgf push!(a, Plot(
                   {
                    only_marks,
                    color = "red",
                    mark = "x",
                   },
                   Table(real.(vals), imag.(vals)))
          );
mx = ceil(Int, maximum(imag.(vals/π)))
@pgf push!(a, Plot(
                   {
                    only_marks,
                    color = "blue",
                    mark = "o",
                   },
                   Table(zeros(2mx + 1), π*(-mx:mx)))
          );
pgfsave("spectra.pdf", a)

vals = ev.values[findall(real.(ev.values) .> -0.1)]
vals = vals[findall(imag.(vals) .≥ 0)]
p = sortperm(imag.(vals[findall(imag.(vals) .≥ 0)]))
vals = vals[p]
@pgf a = Axis(
              {
              }
             );
@pgf push!(a, Plot(
                   {
                    only_marks,
                    color = "red",
                    mark = "x",
                   },
                   Table(1:length(vals), imag.(vals)))
          );
@pgf push!(a, Plot(
                   {
                    only_marks,
                    color = "black",
                    mark = "x",
                   },
                   Table(1:length(vals), -1000*real.(vals) .+ π*(0:length(vals)-1)))
          );
@pgf push!(a, Plot(
                   {
                    only_marks,
                    color = "blue",
                    mark = "o",
                   },
                   Table(1:length(vals), π*(0:length(vals)-1)))
          );
pgfsave("spectra.pdf", a)
