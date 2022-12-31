initime = time()
using LinearAlgebra, Distributions, Random, Statistics
# BLAS.set_num_threads(1)
# ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
# using DataFrames
using DelimitedFiles, Plots, StatsPlots, LaTeXStrings
Threads.nthreads()

Random.seed!(7)
n = 10^5
d = 51
beta0 = ones(d)
ds = d - 1
corr  = 0.5
sigmax = [corr+(1-corr)*(i==j) for i in 1:ds, j in 1:ds]
nmd = 5
case = haskey(ENV, "case") ? parse(Int, ENV["case"]) : 3
case_ss = [1, 2, 3, 4, 5, 6]
lcs = length(case_ss)
k = 5000
k_ss = [2000, 3000, 5000, 10000, 20000, 50000][4]
lks = length(k_ss)
for k in k_ss
    for case in case_ss
        Random.seed!(7)
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

        # PI_poi = Array{Float64}(undef, n, lks)
        # for (ik, k) in enumerate(k_ss)
        idxH = sum(csh .> (k.-collect(n:-1:1).+1) .* sh)
        g = n - idxH
        H = csh[idxH] / (k-g)
        PIp = min.(h, H) ./ sum(min.(h, H))
        # PI_poi[:,ik] = PIp

        lim = (PI[1], PI[end]).*n
        p_wrp = histogram(PI.*n, legend=false, normalize=:probability,
                          xlim=lim,linecolor=:red, fillcolor=:red);
        xlabel!(L"\mathbf{ \pi_{Ri}^{opt} }", xguidefontsize=10);
        p_poi = histogram(PIp.*n, legend=false, normalize=:probability,
                          xlim=lim,linecolor=:red, fillcolor=:red);
        xlabel!(L"\mathbf{ \pi_{Pi}^{opt} }", xguidefontsize=10);
        ps = plot(PI.*n, PIp.*n, seriestype=:scatter, xlim=lim, ylim=lim,
                  legend=false);
        xlabel!(L"\mathbf{ \pi_{Ri}^{opt} }", xguidefontsize=10);
        ylabel!(L"\mathbf{ \pi_{Pi}^{opt} }", yguidefontsize=10);
        plot!(PI.*n, PI.*n, legend=false);
        l = @layout [[a; b] c]
        pall = plot(p_wrp, p_poi, ps, layout=l, size=(200*2.5, 200));
        fn = "output/PIDiffCase-k$(k)-c$(case).png"
        savefig(pall, fn)
        # end
    end
    pngs = "output/PIDiffCase-k$(k)-c" .* map(string, case_ss) .* ".png"
    run(`convert $(pngs) output/0PIDiffCase-k$(k).pdf`)
    run(`cp output/0PIDiffCase-k$(k).pdf $(homedir())/Dropbox/work/OSMAC/general/draft/figures/0PIDiffCase-k$(k).pdf`)
end
print(time() - initime)
