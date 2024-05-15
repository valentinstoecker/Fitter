### A Pluto.jl notebook ###
# v0.19.35

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 40bbb7f0-0623-11ef-23fb-6da986d3325a
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()

using HZBTools, Base64, Plots, Statistics, ForwardDiff, Optimization,
  OptimizationOptimJL, LinearAlgebra, FFTW, CSV, DataFrames, PlutoUI, HDF5
md"""Package management"""
end

# ╔═╡ 9a05f405-0ea7-4512-8554-4797f9a34be3
md"""
# Data
$(@bind file FilePicker())
"""

# ╔═╡ b932b9f7-afe5-4a40-be11-bd8472cb973e
md"""
x-column: $(@bind column_x PlutoUI.NumberField(1:20))
y-column: $(@bind column_y PlutoUI.NumberField(1:20))
z-column: $(@bind column_z PlutoUI.NumberField(1:20))
"""

# ╔═╡ f9062d99-d601-401b-a5a5-449b4c3c3dfa
begin
x, y, z = HZBTools.import_file_2d(String(copy(file["data"])),column_x, column_y, column_z)
md""""""
end

# ╔═╡ f859b2fb-e9b2-4b71-8dba-659633d242b4
md"""
x-start: $(@bind x_start NumberField(LinRange(x[1], x[end], length(x)), default=x[1])) \
x-end: $(@bind x_end NumberField(LinRange(x[1], x[end], length(x)), default=x[end])) \
y-start: $(@bind y_start NumberField(LinRange(y[1], y[end], length(y)), default=y[1])) \
y-end: $(@bind y_end NumberField(LinRange(y[1], y[end], length(y)), default=y[end]))
"""

# ╔═╡ a9ebe0be-fa41-416d-8331-036e354b7e1b
begin
	function to_index(xs, f)
		round(Int, (f-xs[1])/(xs[2]-xs[1])) + 1
	end
	
	x_trimmed = x[to_index(x, x_start):to_index(x, x_end)]
	y_trimmed = y[to_index(y, y_start):to_index(y, y_end)]
	z_trimmed = z[to_index(x, x_start):to_index(x, x_end), to_index(y, y_start):to_index(y, y_end)]
	heatmap(x_trimmed, y_trimmed, z_trimmed', fill=true)
end

# ╔═╡ 685c3861-b56f-493b-80cd-fd0ec4ae2111
begin
	plt = plot(legend=false)
	for i ∈ 1:length(y_trimmed)
		plot!(x_trimmed, z_trimmed[:,i], title="scan lines")
	end
	plt
end

# ╔═╡ 2db00e90-9460-449a-9ebb-479cdbf0f5c3
begin
	z_mean = mean(z_trimmed, dims=2)
	plot(x_trimmed, z_mean, legend=false, title="avg. of scan lines")
end

# ╔═╡ d9c901a4-dcd3-4975-a954-afed7a9ea971
md"""PSD window count: $(@bind psd_windows NumberField(1:16, default=4))"""

# ╔═╡ abeee6e0-a0ed-4f54-890f-a4afa46f0e3c
begin
	z_psd, x_psd = HZBTools.psd_windowed(z_mean
, x_trimmed[2]-x_trimmed[1], psd_windows)
	plot(x_psd,z_psd, xscale=:log10, yscale=:log10, legend=false, title="PSD of averages")
end

# ╔═╡ ade3e860-0c72-4a7f-a72d-b3ef8803a7f6
begin
	HZBTools.downloadDF(DataFrame(x=vec(x_psd), y=vec(z_psd)), file["name"]*"_psd_of_average")
end

# ╔═╡ c2ae4019-86b6-417c-a20f-55717d810478
begin
	x2_psd = Matrix{Float64}(undef, 0, 0)
	zs_psd = Matrix{Float64}(undef, 0, length(x_psd))
	for i ∈ 1:length(y_trimmed)
		nz_psd, x2_psd = HZBTools.psd_windowed(z_trimmed[:,i], x_trimmed[2]-x_trimmed[1], psd_windows)
		zs_psd = vcat(zs_psd,nz_psd')
	end
	p = plot(xscale=:log10, yscale=:log10, legend=false, title="PSDs of scan lines")
	for i ∈ 1:length(y_trimmed)
		plot!(x2_psd, zs_psd[i,:])
	end
	p
end

# ╔═╡ 9651fb56-8d76-4132-b205-42593d64f424
plot(x2_psd, mean(zs_psd, dims=1)', xscale=:log10, yscale=:log10, legend=false, title="avg. of PSDs")

# ╔═╡ 12065a65-267b-4918-8cab-31e47c34f5f5
begin
	HZBTools.downloadDF(DataFrame(x=vec(x2_psd), y=vec(mean(zs_psd, dims=1)')), file["name"]*"_average_psd")
end

# ╔═╡ Cell order:
# ╟─9a05f405-0ea7-4512-8554-4797f9a34be3
# ╟─b932b9f7-afe5-4a40-be11-bd8472cb973e
# ╟─f9062d99-d601-401b-a5a5-449b4c3c3dfa
# ╟─f859b2fb-e9b2-4b71-8dba-659633d242b4
# ╟─a9ebe0be-fa41-416d-8331-036e354b7e1b
# ╟─685c3861-b56f-493b-80cd-fd0ec4ae2111
# ╟─2db00e90-9460-449a-9ebb-479cdbf0f5c3
# ╟─d9c901a4-dcd3-4975-a954-afed7a9ea971
# ╟─abeee6e0-a0ed-4f54-890f-a4afa46f0e3c
# ╟─ade3e860-0c72-4a7f-a72d-b3ef8803a7f6
# ╟─c2ae4019-86b6-417c-a20f-55717d810478
# ╟─9651fb56-8d76-4132-b205-42593d64f424
# ╟─12065a65-267b-4918-8cab-31e47c34f5f5
# ╟─40bbb7f0-0623-11ef-23fb-6da986d3325a
