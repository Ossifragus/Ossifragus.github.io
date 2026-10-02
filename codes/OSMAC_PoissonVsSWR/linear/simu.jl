initime = time()
# export JULIA_NUM_THREADS=4
# Threads.nthreads()
using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(1)
# ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
using Distributions, Plots, LaTeXStrings, Feather
# using DataFrames
using DelimitedFiles
include("gendat.jl")

function findH(x, rho)
    # rho = 0.6
    n = length(x)
    s = n * rho
    sm = sum(x)
    h = x[end]
    g = 0
    while (sm < (s-g) * h) & (g <= s)
        # global sm, g, h
        sm -= x[n-g]
        g += 1
        h = x[n-g]
    end
    return (h, g)
end

function simu(X::Matrix, Y::Vector, ks::Array{Int,1}, rpt::Int=10, nmd::Int=2; k0=1000)
    (n,d) = size(X)
    lks = length(ks)
    Betas = fill(NaN, d, rpt, nmd, lks)

    # k0=1000
    a = 0.1
    
    idx_plt = sample(1:n, k0, replace=true)
    x_plt = X[idx_plt, :]
    y_plt = Y[idx_plt]
    ddm_plt = (x_plt'x_plt) 
    beta_plt = ddm_plt \ x_plt'*y_plt
    e_plt = Y .- X * beta_plt
    dm = abs.(e_plt) .* sqrt.(vec(sum(X.^2, dims=2)))

    pi_R = (1-a) .* dm ./ sum(dm) .+ a/n
    pi_P = (1-a) .* dm ./ (n*mean(dm[idx_plt])) .+ a/n


    for (ik, k) in enumerate(ks)
        print("k=$(k)\n")

        # oH = findH(dm[idx_plt], (k-k0)/n)[1]
        # dmh = min.(dm, oH) # dm # 
        # dmh = min.(dm, maximum(dm[idx_plt]))
        dmh = min.(dm, quantile(dm[idx_plt], 1-k/5n))
        pi_PH = (1-a) .* dmh ./ (n*mean(dmh[idx_plt])) .+ a/n

        for rr in 1:rpt
            # if mod(rr, 10) ==0
            #     print("$rr ")
            # end
            # R opt
            idx_ropt = wsample(1:n, pi_R, k-k0, replace=true)
            x_ropt = X[idx_ropt, :]
            y_ropt = Y[idx_ropt]
            pi_ropt = pi_R[idx_ropt]
            ddm_ropt = (x_ropt' * (x_ropt ./ pi_ropt))
            beta_ropt = ddm_ropt \ (x_ropt ./ pi_ropt)'*y_ropt
            beta_ropt = (ddm_ropt ./n .+ ddm_plt) \ (ddm_ropt*beta_ropt ./n .+ ddm_plt*beta_plt)
            # R uni
            idx_runi = sample(1:n, k, replace=true)
            x_runi = X[idx_runi, :]
            y_runi = Y[idx_runi]
            beta_runi = (x_runi'x_runi) \ x_runi'*y_runi

            # Poisson opt
            u = rand(n)
            idx_popt = u .<= (k-k0).*pi_P
            x_popt = X[idx_popt, :]
            y_popt = Y[idx_popt]
            pi_popt = min.((k-k0) .* pi_P[idx_popt], 1)
            ddm_popt = (x_popt' * (x_popt ./ pi_popt))
            beta_popt = ddm_popt \ (x_popt ./ pi_popt)'*y_popt
            beta_popt = (ddm_popt ./(n/(k-k0)) .+ ddm_plt) \ (ddm_popt*beta_popt ./(n/(k-k0)) .+ ddm_plt*beta_plt)

            # Poisson opt est H
            idx_popth = u .<= (k-k0).*pi_PH
            x_popth = X[idx_popth, :]
            y_popth = Y[idx_popth]
            pi_popth = min.((k-k0) .* pi_PH[idx_popth], 1)
            ddm_popth = (x_popth' * (x_popth ./ pi_popth))
            beta_popth = ddm_popth \ (x_popth ./ pi_popth)'*y_popth
            beta_popth = (ddm_popth ./(n/(k-k0)) .+ ddm_plt) \ (ddm_popth*beta_popth ./(n/(k-k0)) .+ ddm_plt*beta_plt)

            idx_puni = u .<= k/n
            x_puni = X[idx_puni, :]
            y_puni = Y[idx_puni]
            beta_puni = (x_puni'x_puni) \ x_puni'*y_puni

            Betas[:,rr,1,ik] = beta_ropt
            Betas[:,rr,2,ik] = beta_runi
            Betas[:,rr,3,ik] = beta_popt
            Betas[:,rr,4,ik] = beta_popth
            Betas[:,rr,5,ik] = beta_puni
        end
    end
    return Betas
end

Random.seed!(1)
n = 10^5
d = 51
beta0 = ones(d)
nmd = 5
case = haskey(ENV, "case") ? parse(Int, ENV["case"]) : 11
rpt = 1000
k = 1000
k0 = 1000
ks = [2000, 5000, 10000, 20000, 50000]

(X, Y, beta_f, lv) = gendat(n, case, beta0)
if case == 11
    alpha = ks ./ n
    n = length(Y)
    ks = convert.(Int, round.(n * alpha, digits=0))
    k0 = convert(Int, round.(ks[1]/2, digits=0))
end

print("case:$(case)\n n:$(n)\n rpt:$(rpt) \n ks:$(ks)\n")
rec = fill(NaN, nmd, length(ks))
@time cal = simu(X, Y, ks, rpt, nmd)
for m in 1:nmd, ik in 1:length(ks)
    rec[m,ik] = sum(mean((cal[:,:,m, ik] .- beta_f).^2, dims=2))
    fname = "output/case$(case)n$(n)method$(m)k$(k).csv"
    writedlm(fname, cal[:,:,m,ik])
end

show(stdout, "text/plain", rec)
print("\n", sum(rec))
# label = ["dc: M=" .* map(string, ks); "uniform"]
label = ["R opt" "R uni" "P opt" "P opt h" "P uni"]
pl = plot(ks/n, log.(rec'), label=label, lw=2, m=(7,:auto))
plot(ks, log.(rec'), label=label, lw=2, m=(7,:auto))
# pl = plot(log10.(ks), log10.(rec'), label=label, lw=2, m=(7,:auto))
# plot(log10.(ks), log10.(rec'), label=label, lw=2, m=(7,:auto))
savefig(pl, "output/case$(case).pdf")
writedlm("output/case$(case).csv", [ks rec'])

print("\n Total time used: $(round(time() - initime)) seconds")

# scatter!(kss, rec', legend=false)
# scatter!(kss, rec', legend=false)

# rec = Array{Float64}(undef, 3, 2)
# # mapslices
# using Statistics
# nanmean(x) = mean(filter(!isnan,x))
# nanmean(x,y) = mapslices(nanmean,x,dims=y)
# y = [NaN 2 3 4;5 6 NaN 8;9 10 11 12]
# nanmean(y)
# nanmean(y,1)
