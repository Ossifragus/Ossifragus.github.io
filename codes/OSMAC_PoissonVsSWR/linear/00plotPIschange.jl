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
case_ss = [1, 2, 3, 4, 5, 6][4]
lcs = length(case_ss)
k = 5000
k_ss = [2000, 3000, 5000, 10000, 20000, 50000]
lks = length(k_ss)
G = Array{Int}(undef, lks, lcs)
# Threads.@threads 
for (icase, case) in enumerate(case_ss)
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

    # k = 5000
    # k_ss = [2000, 5000, 10000, 20000, 50000]
    # lks = length(k_ss)
    PI_poi = Array{Float64}(undef, n, lks)
    for (ik, k) in enumerate(k_ss)
        idxH = sum(csh .> (k.-collect(n:-1:1).+1) .* sh)
        g = n - idxH
        G[ik, icase] = g
        H = csh[idxH] / (k-g)
        PIp = min.(h, H) ./ sum(min.(h, H))
        PI_poi[:,ik] = PIp

        lim = (PI[1], PI[end]).*n
        # p_wrp = histogram(PI.*n, legend=false, normalize=:probability,
        #                   xlim=lim,linecolor=:red, fillcolor=:red);
        p_poi = histogram(PIp.*n, legend=false, normalize=:probability,
                          xlim=lim,linecolor=:red, fillcolor=:red);
        xlabel!(L"\mathbf{\pi_{Pi}^{opt}}", xguidefontsize=10);

        ps = plot(PI.*n, PIp.*n, seriestype=:scatter, xlim=lim, ylim=lim);
        xlabel!(L"\mathbf{ \pi_{Ri}^{opt} }", xguidefontsize=10);
        ylabel!(L"\mathbf{ \pi_{Pi}^{opt} }", yguidefontsize=10);
        plot!(PI .* n, PI .* n, legend=false);
        # l = @layout [a{0.001h}; b c{0.13w}]
        # l = @layout [[a; b] c]
        l = @layout [a b]
        pall = plot(# p_wrp, 
                    p_poi, ps, layout=l, size=(200*2.5, 200));
        fn = "output/PIsChange$(case)-k$(k).png"
        savefig(pall, fn)
    end
pngs = "output/PIsChange$(case)-k" .* map(string, k_ss) .* ".png"
run(`convert $(pngs) output/0PIsChange$(case).pdf`)
run(`cp output/0PIsChange$(case).pdf $(homedir())/Dropbox/work/OSMAC/general/draft/figures/0PIsChange$(case).pdf`)
end

show(stdout, "text/plain", G)
writedlm("output/00numG.csv", G)

print("\n", time() - initime)
# pdfs = "output/0PIsChange" .* map(string, case_ss) .* ".pdf"
# run(`pdftk $(pdfs) output output/00PIsChange.pdf`)

# run(`cp $(pngs) $(homedir())/Dropbox/work/OSMAC/general/draft/figures/`)
# pngs = "output/c" .* map(string, case) .* "-k" .* map(string, k_ss) .* ".png"
# run(`cp $(pngs) $(homedir())/Dropbox/work/OSMAC/general/draft/figures/`)


# run(`pdftk $pdfs output output/0PIc$(case)kchange.pdf`)

# histogram(n*PI, legend=false, normalize=:probability, linecolor=:red)
# xlabel!(L"\mathbf{ \pi_{Ri}^{opt} }", yguidefontsize=15)
# ylabel!(L"\mathbf{ \pi_{Pi}^{opt} }", yguidefontsize=15)
