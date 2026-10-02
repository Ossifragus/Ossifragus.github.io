using Plots, DelimitedFiles, LaTeXStrings
# Plots.scalefontsizes(1.8)
# cd("..")

ncase = [1:3; 10; 11]
n=[repeat([10^5], 3); 581012; 5*10^6]
# Ms = [1, 2, 5, 10, 100]
# label = ["dc: M=" .* map(string, Ms); "uniform"]
label = [L"\mathbf{opt R}" L"\mathbf{uni R}" L"\mathbf{opt P}_\infty" #=
         =# L"\mathbf{opt P}_{b=3}" L"\mathbf{uni P}"]

rec = Array{Float64}(undef, 5, 6, 5)
for (i, case) in enumerate(ncase)
    rs = readdlm("output/case$(case).csv")
    rec[:,:,i] = rs
end

    pl = plot(rs[1:end,1]/n[i], log.(rs[1:end,2:end]),
              label=label, lw=3, m=(10,:auto),
              tickfontsize=18, # legend=:top,
              legendfontsize=15,
              legend= case == 0 ? true : false
              )
    # xlabel!(L"\mathbf{{(s_0+s)}/{n}}")
    # ylabel!("log(MSE)")
    # savefig(pl, "output/case$(case).pdf")

lg = plot(ones(1, length(ncase)), label=label, lw=3, m=(0,:auto),
          showaxis=false, grid=false,
          legend=:inside, legendfontsize=32)
scatter!(ones(1, 1), color=:white,
         markerstrokecolor=:white, m=(0), label="")
savefig(lg, "output/casel.pdf")
ncase = [ncase; "l"]

pdfs = "output/case" .* map(string, ncase) .* ".pdf"
run(`pdftk $pdfs output mse_logistic.pdf`)
# run(`pdftk $pdfs output $(homedir())/Dropbox/work/OSMAC/general/m-estimation/icml2019/figures/mse_logistic.pdf`)

#
# rs = readdlm("output/case1.csv")
