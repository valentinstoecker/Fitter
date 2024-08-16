### A Pluto.jl notebook ###
# v0.19.42

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

# ╔═╡ cd186284-fcb4-11ee-3829-c5cd8418f889
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

# ╔═╡ d239df4b-23c0-4d19-8064-97d6290e0797
md"""
# Data
$(@bind file FilePicker())
"""

# ╔═╡ 1e43bbfd-edbf-4b9d-bceb-7837848ff43e
md"""
x-column: $(@bind column_x PlutoUI.NumberField(1:10))
y-column: $(@bind column_y PlutoUI.NumberField(1:10))
"""

# ╔═╡ 534352c9-9683-465d-8c84-7aa21362d22a
begin
	data = HZBTools.import_file(String(copy(file["data"])), column_x, column_y)
	x_data, y_data = data[:,1], data[:,2]
	x_min = minimum(x_data)
	x_max = maximum(x_data)
	plot(x_data,y_data, xlabel="x in mm", ylabel="slope in arcsec", label="slope")
end

# ╔═╡ 2d495998-ef4b-4622-a9b6-b2087861066e
md"""
x-start: $(@bind x_start NumberField(LinRange(x_min, x_max, length(x_data)), default=x_min))
x-end: $(@bind x_end NumberField(LinRange(x_min, x_max, length(x_data)),default=x_max))
"""

# ╔═╡ 2f470b07-d941-4a91-8f80-2008a4613e27
begin
	start_index = findfirst(i -> x_data[i] == x_start, 1:length(x_data))
	end_index = findfirst(i -> x_data[i] == x_end, 1:length(x_data))
	x_trimmed = x_data[start_index:end_index]
	y_trimmed = y_data[start_index:end_index]
	plot(x_trimmed, y_trimmed, xlabel="x in mm", ylabel="slope in arcsec",label="slope")
	y_integrated = HZBTools.integrate(x_trimmed,  HZBTools.arcsec2slope.(-y_trimmed))
	plot!(twinx(), x_trimmed, y_integrated, ylabel="y in mm", color=:red,label="y")
end

# ╔═╡ e0b06662-c7e3-44c0-95e8-309125e85009
md"""
## Fit
curve-type: $(@bind curve_type Select(["ellipse", "parabola", "polynomial"]))
"""

# ╔═╡ 807d8f86-4c2a-4acf-8a0f-e89399c9328c
if curve_type == "ellipse"
	md"""
	R in mm: $(@bind R_text TextField()) ΔR in mm: $(@bind ΔR_text TextField())\
	r in mm: $(@bind r_text TextField()) Δr in mm: $(@bind Δr_text TextField())\
	θ in °: $(@bind θ_text TextField()) Δθ in °: $(@bind Δθ_text TextField())\
	Δα in °: $(@bind Δα_text TextField()) calculate: $(@bind calculate CheckBox())
	"""
elseif curve_type == "parabola"
	md"""
	l in mm: $(@bind l_text TextField()) Δl in mm: $(@bind ΔR_text TextField())\
	θ in °: $(@bind θ_text TextField()) Δθ in °: $(@bind ΔR_text TextField())\
	Δα in °: $(@bind Δα_text TextField()) calculate: $(@bind calculate CheckBox())
	"""
elseif curve_type == "polynomial"
	md"""
	order: $(@bind p_order TextField())
	"""
end

# ╔═╡ 893c06e1-a7ef-4159-aee8-c637ef4ba939
md"""
## Residual
"""

# ╔═╡ f39b8547-566d-443a-bdaa-d60aaa97a6d8
md"""
## PSD of residual
window count: $(@bind window_count NumberField(1:100))
logarithmic x: $(@bind x_log CheckBox(default=true))
"""

# ╔═╡ 206c8337-14bf-4ca7-9d3c-3f51a31e4c61
if curve_type == "polynomial"
	md"""
	### Polynomial
	
	#### Coefficients
	
	The coefficients describe the fitted polynomial where x = 0 is the center of the measured section. The first element of the coefficient list is the factor of the term with the highest order.
	"""
end

# ╔═╡ 5acdb46e-8984-454c-8d8b-f0b398628527
if curve_type == "polynomial"
	md"""
	#### Excel Formula
	
	cell name: $(@bind cell_ref TextField())
	
	decimal seperator: $(@bind ds Select([", (comma)", ". (dot)"]))
	"""
end

# ╔═╡ 9383cecf-dcfc-4121-a6b9-0eac29934502
if curve_type == "polynomial"
	md"""
	coordinate system: $(@bind coord_system Select(["origin", "original"]))
	"""
end

# ╔═╡ 51a39cb8-c130-4f8a-9789-8e4bb358353d
begin
	max_digits = 7
	max_time = 10
	md"""
	config
	"""
end

# ╔═╡ 322367d8-7b04-42bc-9661-28fbc549e68d
begin
	function updateΔ(xs)
		map(x -> x == 0 ? 1e-10 : x, xs)
	end
	if curve_type == "ellipse" && calculate
		R = parse(Float64, R_text)
		r = parse(Float64, r_text)
		θ = parse(Float64, θ_text)
		ΔR = parse(Float64, ΔR_text)
		Δr = parse(Float64, Δr_text)
		Δθ = parse(Float64, Δθ_text)
		Δα = parse(Float64, Δα_text)
		x = x_trimmed .- mean(x_trimmed)
		y = -y_trimmed
		fₑ(cs) = mean((HZBTools.ell_from_p(cs).(x) - y).^2)
		u0 = [R, r,deg2rad(θ), HZBTools.arcsec2rad(y[div(length(y),2)])]
		ϵ_u = updateΔ([ΔR, Δr, deg2rad(Δθ), deg2rad(Δα)])
		f_opt = OptimizationFunction((u,p) ->fₑ(u), AutoForwardDiff())
		prob = Optimization.OptimizationProblem(f_opt, u0; lb=u0-ϵ_u,ub=u0+ϵ_u)
		opt_res = solve(prob,  BFGS();  maxtime = max_time)
		sol = opt_res.u
		R_fit = round(sol[1], digits=max_digits)
		r_fit = round(sol[2], digits=max_digits)
		θ_fit = round(rad2deg(sol[3]), digits=max_digits)
		α_fit = round(rad2deg(sol[4]), digits=max_digits)
		xₑ = HZBTools.ell_from_p([R_fit,r_fit,deg2rad(θ_fit),deg2rad(α_fit)]).(x_trimmed .- mean(x_trimmed)) + y_trimmed
		md"""
		Fit: \
		R: $(R_fit) mm r: $(r_fit) mm θ: $(θ_fit) ° α: $(α_fit) °\
		rms: $(√(mean(xₑ.^2)))
		"""
	elseif curve_type == "parabola" && calculate
		l = parse(Float64, l_text)
		θ = parse(Float64, θ_text)
		Δl = parse(Float64, Δl_text)
		Δθ = parse(Float64, Δθ_text)
		Δα = parse(Float64, Δα_text)
		x = x_trimmed .- mean(x_trimmed)
		y = y_trimmed
		fₑ(cs) = mean((HZBTools.par_from_p(cs).(x) - y).^2)
		u0 = [l,deg2rad(θ), HZBTools.arcsec2rad(y[div(length(y),2)])]
		ϵ_u = updateΔ([Δl, deg2rad(Δθ), deg2rad(Δα)])
		f_opt = OptimizationFunction((u,p) ->fₑ(u), AutoForwardDiff())
		prob = Optimization.OptimizationProblem(f_opt, u0; lb=u0-ϵ_u,ub=u0+ϵ_u)
		opt_res = solve(prob,  BFGS();  maxtime = max_time)
		sol = opt_res.u
		l_fit = round(sol[1], digits=max_digits)
		θ_fit = round(rad2deg(sol[2]), digits=max_digits)
		α_fit = round(rad2deg(sol[3]), digits=max_digits)
		xₑ = HZBTools.par_from_p([l,deg2rad(θ),deg2rad(α)]).(x_trimmed .- mean(x_trimmed)) + y_trimmed
		md"""
		Fit: \
		l: $(l_fit) mm θ: $(θ_fit) ° α: $(α_fit) °\
		rms: $(√(mean(xₑ.^2)))
		"""
	elseif curve_type == "polynomial"
		n = parse(Int64, p_order)
		x = x_trimmed .- mean(x_trimmed)
		y = y_trimmed
		A = zeros(length(x_trimmed), n + 1)
		for i ∈ 1:length(x), j ∈ 1:(n+1)
			A[i, j] = x[i]^(j - 1)
		end
		coeffs = (A' * A)\(A'*y)
		f_poly(x) = evalpoly(x, coeffs)
		y_poly = f_poly.(x)
		xₑ = y_trimmed - y_poly
		md"""
		rms: $(√(mean(xₑ.^2)))
		"""
	end
end

# ╔═╡ 75fea479-2b72-4025-aef1-3c3c39e12479
begin
plot(x_trimmed, xₑ, xlabel="x in mm", ylabel="Δslope in arcsec", label="Δslope")
xₑ_integrated = HZBTools.integrate(x_trimmed,  HZBTools.arcsec2slope.(-xₑ))
plot!(twinx(),x_trimmed,xₑ_integrated, ylabel="Δy in mm", label="Δy", color=:red)
end

# ╔═╡ 7bdcc3a2-3082-4b16-b62b-20ecc013d6b3
begin
	Δt = x_trimmed[2] - x_trimmed[1]
  	y_psd, x_psd = HZBTools.psd_windowed(xₑ, 1/Δt, window_count)
	plot(x_psd, y_psd, xscale=(x_log ? :log10 : :identity), yscale=:log10)
end

# ╔═╡ d14d40f9-27ee-493d-b854-e59fd2f5a2d9
filename = split(file["name"],".")[1]

# ╔═╡ a74dd0e5-67bf-485c-bb59-9b345d0cddb4
begin
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

	# Kahan summation algorithm
	function Base.sum(xs :: AbstractArray{Float64})
		s = 0
		c = 0
		for x ∈ xs
			y = x - c
			t = s + y
			c = (t - s) - y
			s = t
		end
		return s
	end

	function translatePoly(p, o)
		n = length(p) - 1
		p_n = zeros(length(p))
		for k ∈ 0:n
			cs = zeros(n - k + 1)
			for i ∈ k:n
				cᵢ = p[i + 1]
				cs[i - k + 1] = binomial(i,k)*cᵢ*o^(i-k)
			end
			p_n[k + 1] = sum(cs)
		end
		p_n
	end

	function coeffsToExcel(coeffs, cell = "A1", offset = 0)
		s = "0"
		for c ∈ reverse(coeffs)
			s = "(($cell + $offset)*$s + $c)"
		end
		return s
	end
	
	md"utils"
end

# ╔═╡ d664cb73-c260-4d3b-b654-358d1e3f0bc0
begin
	residual_df = DataFrame(x=x_trimmed, residual=xₑ, residual_integrated=xₑ_integrated)
	downloadDF(residual_df, "$(filename)_residual")
end

# ╔═╡ e7c68ba4-c602-4198-9797-9aa7bb2c58d6
begin
	psd_df = DataFrame(f=collect(x_psd), amplitude=y_psd[:,1])
	downloadDF(psd_df, "$(filename)_psd_$(window_count)_windows")
end

# ╔═╡ f4b890ff-861b-4bda-a545-7b958d8b1691
if curve_type == "polynomial"
	df = DataFrame(coefficients=collect(reverse(coeffs)))
	downloadDF(df, "$(filename)_polynomial_$(n)th-order")
end

# ╔═╡ 8b4665f6-9604-4418-b123-fe221f33ba9e
if curve_type == "polynomial"
	offset = 0
	if coord_system == "original"
		offset = -mean(x_trimmed)
	end
	s = coeffsToExcel(coeffs, cell_ref, offset)
	if ds == ", (comma)"
		s = replace(s, "." => ",", "," => ".")
	end
	md"Formula: =$s"
end

# ╔═╡ Cell order:
# ╟─d239df4b-23c0-4d19-8064-97d6290e0797
# ╟─1e43bbfd-edbf-4b9d-bceb-7837848ff43e
# ╟─534352c9-9683-465d-8c84-7aa21362d22a
# ╟─2d495998-ef4b-4622-a9b6-b2087861066e
# ╟─2f470b07-d941-4a91-8f80-2008a4613e27
# ╟─e0b06662-c7e3-44c0-95e8-309125e85009
# ╟─807d8f86-4c2a-4acf-8a0f-e89399c9328c
# ╟─322367d8-7b04-42bc-9661-28fbc549e68d
# ╟─893c06e1-a7ef-4159-aee8-c637ef4ba939
# ╟─75fea479-2b72-4025-aef1-3c3c39e12479
# ╟─d664cb73-c260-4d3b-b654-358d1e3f0bc0
# ╟─f39b8547-566d-443a-bdaa-d60aaa97a6d8
# ╟─7bdcc3a2-3082-4b16-b62b-20ecc013d6b3
# ╟─e7c68ba4-c602-4198-9797-9aa7bb2c58d6
# ╟─206c8337-14bf-4ca7-9d3c-3f51a31e4c61
# ╟─f4b890ff-861b-4bda-a545-7b958d8b1691
# ╟─5acdb46e-8984-454c-8d8b-f0b398628527
# ╟─9383cecf-dcfc-4121-a6b9-0eac29934502
# ╟─8b4665f6-9604-4418-b123-fe221f33ba9e
# ╟─cd186284-fcb4-11ee-3829-c5cd8418f889
# ╟─51a39cb8-c130-4f8a-9789-8e4bb358353d
# ╟─d14d40f9-27ee-493d-b854-e59fd2f5a2d9
# ╟─a74dd0e5-67bf-485c-bb59-9b345d0cddb4
