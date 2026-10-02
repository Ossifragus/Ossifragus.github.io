initime = time()
using LinearAlgebra, Distributions, Random, Statistics
# BLAS.set_num_threads(1)
# ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
# using DataFrames
using DelimitedFiles, Plots, StatsPlots, LaTeXStrings
# include("gendat.jl")

Random.seed!(1)
n = 10^5
d = 51
beta0 = ones(d)
ds = d - 1
corr  = 0.5
sigmax = [corr+(1-corr)*(i==j) for i in 1:ds, j in 1:ds]
nmd = 5
case = haskey(ENV, "case") ? parse(Int, ENV["case"]) : 3
case_ss = collect(1:6)
lcs = length(case_ss)
# G = Array{Float64}(undef, , lks)
for case in case_ss
    if case == 1 # Normal
        Z = collect(rand(MvNormal(zeros(ds), sigmax), n)')
    else # T
        df = [5, 4, 3, 2, 1][case-1]
        Z = rand(MvNormal(zeros(ds), sigmax), n);
        Z = collect(Z') ./ sqrt.(rand(Chisq(df), n)./df)
    end
    Y = beta0[1] .+ Z * beta0[2:end] + randn(n)
    X = [ones(n) Z]
    F = svd(X)
    beta_f = F.V * Diagonal(1 ./ F.S) * F.U' * Y
    lv = vec(sum(F.U.^2, dims=2))
    e = Y - X * beta_f
    h = abs.(e) .* sqrt.(lv)
    sort!(h)
    PI = h ./ sum(h)

    lvr = sqrt.(lv) ./ sum(sqrt.(lv))

    sh = sort(h)
    csh = accumulate(+, sh)

    k = 5000
    ks = [2000, 5000, 10000, 20000, 50000]
    lks = length(ks)
    PI_poi = Array{Float64}(undef, n, lks)
    for (ik, k) in enumerate(ks)
        idxH = sum(csh .> (k.-collect(n:-1:1).+1) .* sh)
        g = n - idxH
        H = csh[idxH] / (k-g)
        PIp = min.(h, H) ./ sum(min.(h, H))
        PI_poi[:,ik] = PIp

        p_wrp = histogram(PI, legend=false, normalize=:probability);
        p_poi = histogram(PIp, legend=false, normalize=:probability);
        lim = (PI[1], PI[end])
        ps = plot(PI, PIp, seriestype=:scatter, xlim=lim, ylim=lim);
        plot!(PI, PI, legend=false);
        # l = @layout [a{0.001h}; b c{0.13w}]
        l = @layout [[a; b] c]
        pall = plot(p_wrp, p_poi, ps, layout=l, size=(800, 400), title="c$(case)-k$(k).png");
        fn = "c$(case)-k$(k).png"
        savefig(pall, fn)

    end

    idxH = sum(csh .> (k.-collect(n:-1:1).+1) .* sh)
    g = n - idxH
    H = csh[idxH] / (k-g)
    PIp = min.(h, H) ./ sum(min.(h, H))


    # p_wrp = histogram(PI, legend=false, axis=false, grid=false, orientation=:horizontal)
    # p_poi = histogram(PIp, legend=false, axis=false, grid=false, orientation=:horizontal)
    # using RCall
    # @rput PI
    # R"hist(PI)"


    # ps = plot(PI, PI_poi, seriestype=:scatter, legend=:left)

    # plot(PI, PI_poi, seriestype=:scatter, xlim=lim, ylim=lim)

    # plot!(PI[(n-g+1):n], PIp[(n-g+1):n], seriestype=:scatter)
    # plot!(PI[1:idxH], PI[1:idxH])

    # cornerplot([PI PIp])

    # csh[idxH] > (k-g) * sh[idxH]
    # csh[idxH+1] > (k-(g-1)) * sh[idxH+1]
    # H = csh[idxH] / (k-g)
    # sh[n-g]
    # sh[n-g+1]
    # H > sh[n-g]
    # H <= sh[n-g+1]
    # findH = function(sh::Vector, csh::Vector, k::Int=10)
    #     n = length(sh)
    #     for i in n:-1:(n-k)
    #         csh[i] (k-(n-i+1)+1) * sh[i]
    #     end
    #     return 
    # end



    # lv = sqrt.(vec(sum(X.^2, dims=2)))
    # lv = lv ./ sum(lv)
    #=

    plot(lv)

    simu = function(X::Matrix, Y::Vector, ks::Array{Int,1}, rpt::Int=10, nmd::Int=2)
    (n,d) = size(X)
    lks = length(ks)
    Betas = fill(NaN, d, rpt, nmd, lks)

    k0=1000
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

    dmh = min.(dm, quantile(dm, 1-k/3n))
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
    case = haskey(ENV, "case") ? parse(Int, ENV["case"]) : 3
    rpt = 1000
    k = 1000
    ks = [2000, 5000, 10000, 20000, 50000]

    (X, Y, beta_f, lv) = gendat(n, case, beta0)

    print("case:$(case)\n n:$(n)\n rpt:$(rpt) \n ks:$(ks)\n")
    rec = fill(NaN, nmd, length(ks))
    @time cal = simu(X, Y, ks, rpt, nmd)
    for m in 1:nmd, ik in 1:length(ks)
    rec[m,ik] = sum(mean((cal[:,:,m, ik] .- beta_f).^2, dims=2))
    fname = "output/case$(case)n$(n)method$(m)k$(k).csv"
    writedlm(fname, cal[:,:,m,ik])
    end

    # show(stdout, "text/plain", rec)
    # print("\n", sum(rec))
    # # label = ["dc: M=" .* map(string, ks); "uniform"]
    # label = ["R opt", "R uni", "P opt", "P opt h", "P uni"]
    # pl = plot(ks/n, log.(rec'), label=label, lw=2, m=(7,:auto))
    # plot(ks, log.(rec'), label=label, lw=2, m=(7,:auto))
    # # pl = plot(log10.(ks), log10.(rec'), label=label, lw=2, m=(7,:auto))
    # # plot(log10.(ks), log10.(rec'), label=label, lw=2, m=(7,:auto))
    # savefig(pl, "output/case$(case).pdf")
    # writedlm("output/case$(case).csv", [ks rec'])

    # print("\n Total time used: $(round(time() - initime)) seconds")

    # # scatter!(kss, rec', legend=false)
    # # scatter!(kss, rec', legend=false)

    # # rec = Array{Float64}(undef, 3, 2)
    # # # mapslices
    # # using Statistics
    # # nanmean(x) = mean(filter(!isnan,x))
    # # nanmean(x,y) = mapslices(nanmean,x,dims=y)
    # # y = [NaN 2 3 4;5 6 NaN 8;9 10 11 12]
    # # nanmean(y)
    # # nanmean(y,1)
    =#
end
