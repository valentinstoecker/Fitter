module HZBTools

using Statistics, ForwardDiff, LinearAlgebra, FFTW, HDF5, CSV, DataFrames, PlutoUI, Markdown

function import_file(text, xi, yi)
    lines = split(text, "\n")
    readings = Dict()
    for l ∈ lines
      l = strip(l)
      if length(l) == 0 || startswith(l, "#")
        continue
      end
      vals = split(l)
      x = parse(Float64, vals[xi])
      y = parse(Float64, vals[yi])
      if(haskey(readings,x))
        push!(readings[x], y)
      else
        readings[x] = [y]
      end
    end
    out_data = zeros(length(readings),2)
    for (i, k) ∈ readings |> keys |> collect |> sort |> pairs
      out_data[i,:] = [k, mean(readings[k])]
    end
    out_data
end

function import_file_2d(text, xi, yi, zi)
  lines = split(text, "\n")
  xs = Set{Float64}()
  ys = Set{Float64}()
  for l ∈ lines
    l = strip(l)
    if length(l) == 0 || startswith(l, "#")
      continue
    end
    vals = split(l)
    x = parse(Float64, vals[xi])
    y = parse(Float64, vals[yi])
    xs = union(xs, [x])
    ys = union(ys, [y])
  end
  xs = collect(xs) |> sort
  ys = collect(ys) |> sort
  zs = zeros(length(xs), length(ys))
  x_min, x_step = xs[1], xs[2] - xs[1]
  y_min, y_step = ys[1], ys[2] - ys[1]
  for l ∈ lines
    l = strip(l)
    if length(l) == 0 || startswith(l, "#")
      continue
    end
    vals = split(l)
    x = parse(Float64, vals[xi])
    y = parse(Float64, vals[yi])
    z = parse(Float64, vals[zi])
    i = round(Int, (x - x_min) / x_step) + 1
    j = round(Int, (y - y_min) / y_step) + 1
    zs[i,j] = z
  end
  xs, ys, zs
end
  
δ(f) = x -> ForwardDiff.derivative(f,x)
arcsec2rad(w) = deg2rad(w/3600)
arcsec2slope(w) = tan(arcsec2rad(w))
slope2arcsec(s) = (rad2deg(atan(s))*3600)
  
"""
rot_conic(params, α)

Rotate a conic section defined by the parameters params = (A,B,C,D,E,F) by α.
"""
function rot_conic(params, α)
    s = sin(α)
    c = cos(α)
    [
      c^2     c*s     s^2     0       0       0
      -2*c*s  c^2-s^2 2*s*c   0       0       0
      s^2     -s*c    c^2     0       0       0
      0       0       0       c       s       0
      0       0       0       -s      c       0
      0       0       0       0       0       1
    ]*params
end
  
"""
conic_to_f(params)

Convert a conic section defined by the parameters params = (A,B,C,D,E,F) to a
function y = f(x).
"""
function conic_to_f(params)
    A, B, C, D, E, F = params
    function (x)
        p = (E+B*x)/C
        q = (A*x^2 + D*x + F)/C
        -p/2 - √((p/2)^2 - q)
    end
end
  
"""
ell_from_p(c)

Return derivative of the ellipse defined by the parameters c = (R,r,φ,α).
"""
function ell_from_p(c)
    # ellipse parameters
    R,r,φ,α = c
    F₁ = [-cos(φ-α)*R; sin(φ-α)*R]
    F₂ = [cos(φ+α)*r; sin(φ+α)*r]
    a = (R+r)/2
    v = F₂ - F₁
    θ = angle(v[1]+v[2]im)
    c = norm(v)/2
    M = (F₁+F₂)/2
    b = √(a^2-c^2)
  
    # general conic
    A = a^2*sin(θ)^2 + b^2*cos(θ)^2
    B = 2(b^2 - a^2)*sin(θ)*cos(θ)
    C = a^2*cos(θ)^2 + b^2*sin(θ)^2
    D = -2*A*M[1] - B*M[2]
    E = -B*M[1] -2*C*M[2]
    F = A*M[1]^2 + B*M[1]*M[2] + C*M[2]^2 - a^2*b^2
  
    # solved for y
    df = δ(conic_to_f([A,B,C,D,E,F]))
    x -> slope2arcsec(df(x))
end
  
"""
par_from_p(c)

Return derivative of the parabola defined by the parameters c = (l,θ,α).
"""
function par_from_p(c)
    # parabola parameters
    l,θ,α = c
    β = 2θ - π/2
    a = l*sin(β)
    b = l*cos(β)
    h = (l-a)/2
    x₀= -b
    y₀= -a - h
    c = -y₀/x₀^2
    A = c
    B = 0
    C = 0
    D = -2c*x₀
    E = -1
    F = 0
    df = δ(conic_to_f(rot_conic([A,B,C,D,E,F], α+θ)))
    x -> slope2arcsec(df(x))
end
  
"""
integrate(x,y)

Integrate y(x) using the trapezoid rule.
"""
function integrate(x,y)
    n, = size(x)
    yi = zeros(n)
    for i in 2:n
      yi[i] = yi[i-1] + (y[i] + y[i-1])*(x[i] - x[i-1])/2
    end
    yi
end
  
"""
psd(ys, Δt)

Returns the power spectral density of ys with sampling period Δt.
"""
function psd(ys, Δt)
    n = length(ys)
    Ys = ys |> fft |> fftshift
    fs = fftfreq(n, Δt) |> fftshift
    is = 2:div(n,2)
    (reverse(abs.(Ys[is] .* Ys[n+2 .- is])), reverse(fs[n+2 .- is]))
end
  
"""
psd_windowed(ys, Δt, window_count)

Returns the power spectral density of ys with sampling period Δt. The
power spectral density is calculated using the Welch-method with window_count.
The windows are overlapped by 50%.
"""
function psd_windowed(ys, Δt, window_count = 4)
    n = floor(length(ys) / (window_count/2 + 0.5))
    window = [1 - (2i/(n+1)-1)^2 for i in 1:n]
    runs = []
    f = Nothing
    for i in 1:window_count
      offset = floor(Int, (i-1)*n/2) + 1
      ys_segment = ys[offset:offset+length(window)-1].*window
      p, fs = psd(ys_segment, Δt)
      Wss = n * sum(window .^ 2)
      p = p ./ Wss
      push!(runs, p)
      f = fs
    end
    avg = mean(hcat(runs...),dims=2)
    avg, f
end
  
function downloadDF(df, filename)
  buf = IOBuffer()
  CSV.write(buf, df)
  file_as_bytes = h5open("temp_in_mem", "w"; driver=Drivers.Core(;backing_store=false)) do fid
       for name ∈ names(df)
      fid[name] = df[:, name]
    end
       return Vector{UInt8}(fid)
  end
  button_csv = DownloadButton(take!(buf), "$(filename).csv")
  button_h5 = DownloadButton(file_as_bytes, "$(filename).h5")
  md"""
  $button_csv \
  $button_h5
  """
end

end