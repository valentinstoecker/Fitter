module Fitter

using Dash, Base64, PlotlyJS, Statistics, ForwardDiff, Optimization,
  OptimizationOptimJL, LinearAlgebra, FFTW, CSV, DataFrames

function detectwsl()
    Sys.islinux() &&
    isfile("/proc/sys/kernel/osrelease") &&
    occursin(r"Microsoft|WSL"i, read("/proc/sys/kernel/osrelease", String))
end

function open_in_default_browser(url::AbstractString)::Bool
  try
    if Sys.isapple()
      Base.run(`open $url`)
      return true
    elseif Sys.iswindows() || detectwsl()
      Base.run(`cmd.exe /s /c start "" /b $url`)
      return true
    elseif Sys.islinux()
      browser = "xdg-open"
      if isfile(browser)
        Base.run(`$browser $url`)
        return true
      else
        @warn "Unable to find `xdg-open`. Try `apt install xdg-open`"
        return false
      end
    else
      return false
    end
  catch ex
    return false
  end
end

function import_file(text, data_field, xi, yi)
  split_text = split(text, "#")
  function to_pair(segment)
    parts = split(segment,":",limit=2)
    (strip(parts[1]), strip(parts[2]))
  end
  data = Dict(map(to_pair, split_text[3:end]))
  readings = Dict()
  lines = split(data[data_field], "\n")
  for l ∈ lines
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
  (abs.(Ys[is] .* Ys[n+2 .- is]), fs[n+2 .- is])
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


"""
  Dash.jl layout specification
"""
app = dash(external_stylesheets=["assets/app.css"])

upload = html_div() do
  dcc_upload(
    id="upload-data",
    children=html_div(
      ["Drag and drop or click to select files."],
      className="drag-n-drop"
    )
  ),
  html_div(className="select-grid") do
    html_div("y column: "),
    dcc_input(
      id="data-column",
      type="number",
      value=2,
      min=2,
      step=1,
    )
  end
end

raw_data = html_div() do
  dcc_store(id="raw-data-store"),
  html_h2("raw data average"),
  dcc_graph(id="raw-data")
end

trimmed_data = html_div() do
  dcc_rangeslider(
    id="fit-range",
    tooltip=Dict([:placement => "bottom", :always_visible => true]),
    min=0,
    max=1,
    value=[0, 1],
    step=1,
    marks=Dict([0 => "0", 1 => "1"]),
  ),
  dcc_store(id="trimmed-data-store"),
  html_h2("trimmed data average"),
  dcc_graph(id="trimmed-data")
end

curve_select = html_div() do
  html_h2("curve parameters"),
  html_div(className="select-grid") do
    html_div("fit type: "),
    dcc_dropdown(
      id="fit-type",
      options=[
        (label = "ellipse", value ="ellipse"),
        (label = "parabola", value = "parabola")
      ],
      value="ellipse"
    )
  end
end

ellipse_params = html_div(id="pg-ellipse", className="param-grid") do
  html_div(["parameters"]),
  html_div(["values"]),
  html_div(["tolerances"]),
  html_div(["fit"]),
  html_div("R in mm"),
  dcc_input(id="R", type="number", value=0),
  dcc_input(id="R-diff", type="number", value=0),
  html_div(id="R-value"),
  html_div("r in mm"),
  dcc_input(id="r", type="number", value=0),
  dcc_input(id="r-diff", type="number", value=0),
  html_div(id="r-value"),
  html_div("θ in °"),
  dcc_input(id="theta", type="number", value=0),
  dcc_input(id="theta-diff", type="number", value=0),
  html_div(id="theta-value"),
  html_div("α in °"),
  html_div(),
  dcc_input(id="alpha-diff", type="number", value=0),
  html_div(id="alpha-value"),
  html_button("Fit",id="fit-button", value="Fit", className="fit-button"),
  html_div(id="fit-rms", "rms: 0"),
  dcc_store(id="fitted-parameters-ellipse")
end

parabola_params = html_div(id="pg-parabola", className="param-grid", style=Dict(:display => "none")) do
  html_div(["parameters"]),
  html_div(["values"]),
  html_div(["tolerances"]),
  html_div(["fit"]),
  html_div("l in mm"),
  dcc_input(id="l", type="number", value=0),
  dcc_input(id="l-diff", type="number", value=0),
  html_div(id="l-value"),
  html_div("θ in °"),
  dcc_input(id="p-theta", type="number", value=0),
  dcc_input(id="p-theta-diff", type="number", value=0),
  html_div(id="p-theta-value"),
  html_div("α in °"),
  html_div(),
  dcc_input(id="p-alpha-diff", type="number", value=0),
  html_div(id="p-alpha-value"),
  html_button("Fit",id="p-fit-button", value="Fit", className="fit-button"),
  html_div(id="p-fit-rms", "rms: -"),
  dcc_store(id="fitted-parameters-parabola")
end

residual = html_div() do
  dcc_store(id="residual-data-store"),
  html_h2("residual"),
  dcc_graph(id="residual"),
  html_button("Download Residual",id="download-button-residual",className="fit-button"),
  dcc_download(id="download-residual")
end

power_spectral_density = html_div() do
  html_h2("power spectral density of residual"),
  html_div(className="form-grid") do
    "window count: ",
    dcc_input(id="window-count", type="number", value=4, min=1, step=1),
    "x-axis scale: ",
    dcc_dropdown(
      id="xscale",
      options=[
        (label = "log", value ="log"),
        (label = "linear", value = "linear")
      ],
      value="log"
    )
  end,
  dcc_store(id="psd-data-store"),
  dcc_graph(id="psd"),
  html_button("Download PSD",id="download-button-psd",className="fit-button"),
  dcc_download(id="download-psd")
end

app.layout = html_div(className="content") do
  html_h1("mirror fitter"),
  upload,
  raw_data,
  trimmed_data,
  curve_select,
  ellipse_params,
  parabola_params,
  residual,
  power_spectral_density
end

"""
  Dash.jl callbacks
"""

callback!(
  app,
  Output("raw-data-store", "data"),
  Input("upload-data", "contents"),
  Input("data-column", "value"),
) do contents, column
  x = []
  y = []
  if !(contents isa Nothing)
    text = String(base64decode(split(contents,",")[2]))
    data = import_file(text, "Fit-Info", 1, column)
    x = data[:,1]
    y = data[:,2]
  end
  Dict([:x => x, :y => y])
end

callback!(
  app,
  Output("raw-data", "figure"),
  Input("raw-data-store", "data"),
) do data
  x = data[:x]
  y = data[:y]
  plt = plot(
    [scatter(;x=x, y=y, mode="lines", name="Run Avg")],
    Layout(
      xaxis_title="offset in mm" ,
      yaxis_title="slope in arcsec"
    )
  )
end

callback!(
  app,
  Output("fit-range", "min"),
  Output("fit-range", "max"),
  Output("fit-range", "value"),
  Output("fit-range", "marks"),
  Output("fit-range", "step"),
  Input("raw-data-store", "data"),
) do data
  if data isa Nothing || isempty(data[:x])
    return 0, 1, [0, 1], Dict([0 => "0", 1 => "1"]), 1
  end
  x = data[:x]
  y = data[:y]
  n, = size(x)
  d_min = minimum(x)
  d_max = maximum(x)
  return d_min, d_max, [d_min, d_max], Dict([d_min => string(d_min), d_max => string(d_max)]), (d_max - d_min)/(n-1)
end

callback!(
  app,
  Output("trimmed-data-store", "data"),
  Input("raw-data-store", "data"),
  Input("fit-range", "value"),
) do data, range
  if data isa Nothing || isempty(data[:x])
    return Dict([:x => [], :y => [], :y_int => []])
  end
  x = data[:x]
  y = data[:y]
  d_min, d_max = range
  i_min = findfirst(x .>= d_min)
  i_max = findlast(x .<= d_max)
  xt = x[i_min:i_max]
  yt = y[i_min:i_max]
  n, = size(xt)
  yt_int = integrate(xt, arcsec2slope.(-yt))
  Dict([:x =>xt, :y => yt, :y_int => yt_int])
end

callback!(
  app,
  Output("trimmed-data", "figure"),
  Input("trimmed-data-store", "data"),
) do data
  if data isa Nothing || isempty(data[:x])
    return plot([scatter(;x=[], y=[], mode="lines", name="Run Avg")])
  end
  x = data[:x]
  y = data[:y]
  y_int = data[:y_int]
  d_min = minimum(x)
  d_max = maximum(x)
  n, = size(x)
  plt = plot([
      scatter(;x=x, y=y, mode="lines", name="Run Avg"),
      scatter(;x=x, y=y_int, mode="lines", name="Integrated Run Avg", yaxis="y2")
    ],
    Layout(
      yaxis2=attr(
          overlaying="y",
          side="right"
      ),
      legend=attr(orientation="h"),
      xaxis_title="offset in mm" ,
      yaxis_title="slope in arcsec",
      yaxis2_title="height in mm"
    )
  )
end

callback!(
  app,
  Output("pg-ellipse", "style"),
  Output("pg-parabola", "style"),
  Input("fit-type", "value"),
) do value
  if value == "ellipse"
    return Dict(:display => "grid"), Dict(:display => "none")
  else
    return Dict(:display => "none"), Dict(:display => "grid")
  end
end

callback!(
  app,
  Output("fitted-parameters-ellipse", "data"),
  Input("fit-button", "n_clicks"),
  State("trimmed-data-store", "data"),
  State("R", "value"),
  State("R-diff", "value"),
  State("r", "value"),
  State("r-diff", "value"),
  State("theta", "value"),
  State("theta-diff", "value"),
  State("alpha-diff", "value"),
) do n_clicks, data, R, ΔR, r, Δr, θ, Δθ, Δα
  if data isa Nothing || isempty(data[:x])
    return Dict([:R => 0, :r => 0, :θ => 0, :α => 0])
  end
  xt = data[:x]
  yt = data[:y]
  x = xt .- mean(xt)
  y = -yt
  fₑ(cs) = mean((ell_from_p(cs).(x) - y).^2)
  u0 = [R, r,deg2rad(θ), arcsec2rad(y[div(length(y),2)])]
  ϵ_u = [ΔR, Δr, deg2rad(Δθ), arcsec2rad(Δα)]
  f_opt = OptimizationFunction((u,p) ->fₑ(u), AutoForwardDiff())
  prob = Optimization.OptimizationProblem(f_opt, u0; lb=u0-ϵ_u,ub=u0+ϵ_u)
  opt_res = solve(prob,  BFGS();  maxtime = 10)
  sol = opt_res.u
  Dict([:R => sol[1], :r => sol[2], :θ => rad2deg(sol[3]), :α => rad2deg(sol[4])])
end

callback!(
  app,
  Output("fitted-parameters-parabola", "data"),
  Input("p-fit-button", "n_clicks"),
  State("trimmed-data-store", "data"),
  State("l", "value"),
  State("l-diff", "value"),
  State("p-theta", "value"),
  State("p-theta-diff", "value"),
  State("p-alpha-diff", "value"),
) do n_clicks, data, l, Δl, θ, Δθ, Δα
  if data isa Nothing || isempty(data[:x])
    return Dict([:l => 0, :θ => 0, :α => 0])
  end
  xt = data[:x]
  yt = data[:y]
  x = xt .- mean(xt)
  y = -yt
  fₑ(cs) = mean((par_from_p(cs).(x) - y).^2)
  u0 = [l,deg2rad(θ), arcsec2rad(y[div(length(y),2)])]
  ϵ_u = [Δl, deg2rad(Δθ), deg2rad(Δα)]
  f_opt = OptimizationFunction((u,p) ->fₑ(u), AutoForwardDiff())
  prob = Optimization.OptimizationProblem(f_opt, u0; lb=u0-ϵ_u,ub=u0+ϵ_u)
  opt_res = solve(prob,  BFGS();  maxtime = 10)
  sol = opt_res.u
  Dict([:l => sol[1], :θ => rad2deg(sol[2]), :α => rad2deg(sol[3])])
end

callback!(
  app,
  Output("R-value", "children"),
  Output("r-value", "children"),
  Output("theta-value", "children"),
  Output("alpha-value", "children"),
  Input("fitted-parameters-ellipse", "data"),
) do data
  if data isa Nothing
    return "0", "0", "0", "0"
  end
  R = data[:R]
  r = data[:r]
  θ = data[:θ]
  α = data[:α]
  return string(R), string(r), string(θ), string(α)
end

callback!(
  app,
  Output("l-value", "children"),
  Output("p-theta-value", "children"),
  Output("p-alpha-value", "children"),
  Input("fitted-parameters-parabola", "data"),
) do data
  if data isa Nothing
    return "0", "0", "0"
  end
  l = data[:l]
  θ = data[:θ]
  α = data[:α]
  return string(l), string(θ), string(α)
end

callback!(
  app,
  Output("residual-data-store", "data"),
  Output("fit-rms", "children"),
  Output("p-fit-rms", "children"),
  Input("fitted-parameters-ellipse", "data"),
  Input("fitted-parameters-parabola", "data"),
  Input("fit-type", "value"),
  State("trimmed-data-store", "data")
) do params_e, params_p, ft, data
  if data isa Nothing || isempty(data[:x])
    return (Dict([:x => [], :y => [], :y_int => []]), "rms: -", "rms: -")
  end
  x = data[:x]
  y = -data[:y]
  y_int = data[:y_int]

  if ft == "ellipse"
    if params_e isa Nothing || params_e[:R] == 0
      return (Dict([:x => [], :y => [], :y_int => []]), "rms: -", "rms: -")
    end
    R = params_e[:R]
    r = params_e[:r]
    θ = params_e[:θ]
    α = params_e[:α]
    xₑ = ell_from_p([R,r,deg2rad(θ),deg2rad(α)]).(x .- mean(x)) - y
    Dict([:x => x, :y => xₑ, :y_int => integrate(x, xₑ)]), "rms: $(√(mean(xₑ.^2)))", "rms: -"
  else
    if params_p isa Nothing || params_p[:l] == 0
      return (Dict([:x => [], :y => [], :y_int => []]), "rms: -", "rms: -")
    end
    l = params_p[:l]
    θ = params_p[:θ]
    α = params_p[:α]
    xₑ = par_from_p([l,deg2rad(θ),deg2rad(α)]).(x .- mean(x)) - y
    Dict([:x => x, :y => xₑ, :y_int => integrate(x, xₑ)]), "rms: -", "rms: $(√(mean(xₑ.^2)))"
  end
end

callback!(
  app,
  Output("residual", "figure"),
  Input("residual-data-store", "data"),
) do data
  x = data[:x]
  y = data[:y]
  y_int = data[:y_int]
  plt = plot([
    scatter(;x=x, y=y, mode="lines", name="Residual"),
    scatter(;x=x, y=y_int, mode="lines", name="Integrated Residual", yaxis="y2")
    ],
    Layout(
      yaxis2=attr(
          overlaying="y",
          side="right"
      ),
      legend=attr(orientation="h"),
      xaxis_title="offset in mm" ,
      yaxis_title="residual slope in arcsec",
      yaxis2_title="residual height in mm"
    )
  )
end

callback!(
  app,
  Output("download-residual", "data"),
  Input("download-button-residual", "n_clicks"),
  State("residual-data-store", "data"),
  prevent_initial_call=true,
) do n_clicks, data
  if data isa Nothing || isempty(data[:x])
    return Dict([:content => "", :filename => ""])
  end
  x = data[:x]
  y = data[:y]
  y_int = data[:y_int]
  df = DataFrame(x=x, y=y, y_integrated=y_int)
  csv = CSV.write(IOBuffer(), df, delim='\t')
  Dict([:content => String(take!(csv)), :filename => "residual.csv"])
end

callback!(
  app,
  Output("psd-data-store", "data"),
  Input("residual-data-store", "data"),
  Input("window-count", "value"),
) do data, count
  if data isa Nothing || isempty(data[:x])
    return Dict([:x => [], :y => []])
  end
  x = data[:x]
  y = data[:y]
  Δt = x[2] - x[1]
  y_psd, x_psd = psd_windowed(y, 1/Δt, count)
  Dict([:x => x_psd, :y => y_psd])
end

callback!(
  app,
  Output("psd", "figure"),
  Input("psd-data-store", "data"),
  Input("xscale", "value"),
) do data, xscale
  if data isa Nothing || isempty(data[:x])
    return plot([scatter(;x=[],y=[])])
  end
  x = data[:x]
  y = data[:y]
  plt = plot(
    [scatter(;x=x, y=y, mode="lines", name="PSD")],
    Layout(
      yaxis_type="log",
      xaxis_type=xscale,
      xaxis_title="frequency in mm^-1",
      yaxis_title="power spectral density in arcsec^2*mm"
    )
  )
end

callback!(
  app,
  Output("download-psd", "data"),
  Input("download-button-psd", "n_clicks"),
  State("psd-data-store", "data"),
  State("window-count", "value"),
  prevent_initial_call=true,
) do n_clicks, data, count
  if data isa Nothing || isempty(data[:x])
    return Dict([:content => "", :filename => ""])
  end
  x = data[:x]
  y = data[:y]
  df = DataFrame(frequency=x, psd=y)
  csv = CSV.write(IOBuffer(), df, delim='\t')
  Dict([:content => String(take!(csv)), :filename => "psd-residual-$count-windows.csv"])
end

function open_browser_with_delay(delay=2)
  sleep(delay)
  println("url: http://127.0.0.1:8050")
  open_in_default_browser("http://127.0.0.1:8050")
end

function run()
  Threads.@spawn open_browser_with_delay()
  run_server(app, "127.0.0.1", 8050)
end

end # module Fitter
