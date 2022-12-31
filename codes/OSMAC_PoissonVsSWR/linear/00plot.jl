using Plots, DelimitedFiles, LaTeXStrings
# Plots.scalefontsizes(1.8)
# cd("..")

# using PyPlot, PyCall
ncase = [1:4; 11]
n=[repeat([10^5], 4); 4188261]
# Ms = [1, 2, 5, 10, 100]
# label = ["dc: M=" .* map(string, Ms); "uniform"]
label = [L"\mathbf{opt R}" L"\mathbf{uni R}" L"\mathbf{opt P}_\infty" #=
         =# L"\mathbf{opt P}_{b=5}" L"\mathbf{uni P}"]
for (i, case) in enumerate(ncase)
    rs = readdlm("output/case$(case).csv")
    pl = plot(rs[1:end,1]/n[i], log.(rs[1:end,2:end]),
              label=label, lw=3, m=(10,:auto),
              tickfontsize=18, xguidefontsize=22, yguidefontsize=14,
              legendfontsize=22,
              legend= case == 10 ? true : false
              )
    xlabel!(L"\mathbf{(s_0+s)/n}")
    ylabel!("log(MSE)")
    savefig(pl, "output/case$(case).pdf")
end

# lg = plot(ones(1, ncase+1), label=label, lw=3, m=(10,:auto),
#           showaxis=false, grid=false,
#           legend=:inside, legendfontsize=18)
# scatter!(ones(1, 1), color=:white,
#          markerstrokecolor=:white, m=(20), label="")
# savefig(lg, "output/case$(ncase+1).pdf")

lg = plot(ones(1, length(ncase)), label=label, lw=3, m=(0,:auto),
          showaxis=false, grid=false,
          legend=:inside, legendfontsize=32)
scatter!(ones(1, 1), color=:white,
         markerstrokecolor=:white, m=(0), label="")
savefig(lg, "output/casel.pdf")
ncase = [ncase; "l"]


pdfs = "output/case" .* map(string, ncase) .* ".pdf"
run(`pdftk $pdfs output mse_linear.pdf`)
run(`cp mse_linear.pdf $(homedir())/Dropbox/work/OSMAC/general/draft/figures/`)
# run(`pdftk $pdfs output $(homedir())/Dropbox/work/OSMAC/general/m-estimation/icml2019/figures/mse_linear.pdf`)
# run(`pdftk $pdfs output /Users/haiying/Dropbox/work/OSMAC/general/m-estimation/icml2019/figures/mse_linear.pdf`)


#
# rs = readdlm("output/case1.csv")
