function getMLE(x::Matrix, y::Vector, w::Vector)
    p = size(x)[2]
    beta = zeros(p)
    loop = 1;
    Loop = 100;
    msg = "NA";
    # xt = x'
    while loop <= Loop
        pr = 1 .- 1 ./ (1 .+ exp.(vec(x * beta)))
        H = x' * ((pr .* (1 .- pr) .* w) .* x)
        S = sum((y .- pr) .* w .* x, dims=1)
        shs = try
            H\S'
        catch
            msg = "Not converge";
            println("H is singular")
            beta = NaN * ones(p)
            break
        end
        beta_new = beta .+ shs
        tlr  = sum((beta_new .- beta).^2)
        beta = beta_new
        if tlr < 0.000001
            msg = "Successful convergence"
            break
        end
        if loop == Loop
            msg = "Not converge";
            println("Maximum iteration reached")
            beta = NaN * ones(p)
            break
        end
        loop += 1
    end
    return [beta, msg, loop]
end

function gendat(n::Int, case::Int=1, beta0::Vector=ones(10))
    d = length(beta0)
    ds = d - 1
    corr  = 0.5
    sigmax = [corr+(1-corr)*(i==j) for i in 1:ds, j in 1:ds]
    if case == 1 # Normal
        Z = collect(rand(MvNormal(zeros(ds), sigmax), n)')
    elseif case == 2 # lognormal, inbalanced
        Z = exp.(rand(MvNormal(zeros(ds), sigmax), n)')
    elseif case == 3 # T3
        Z = rand(MvNormal(zeros(ds), sigmax), n);
        Z = collect(Z') ./ sqrt.(rand(Chisq(3), n)./3)

    elseif case == 10 # coytype data
        # dat = readdlm("/home/ossifragus/ondisk/covtype.txt")
        datf = Feather.read("/home/ossifragus/Dropbox/work/ondisk/covtype.feather")
        dat = convert(Array{Float64,2}, datf)
        Y = dat[:,1]
        Z = dat[:,2:end]
        dat = nothing
        n = length(Y)
    elseif case == 11 # susy data
        # dat = readdlm("/home/ossifragus/ondisk/SUSY.txt")
        datf = Feather.read("/home/ossifragus/Dropbox/work/ondisk/SUSY.feather")
        dat = convert(Array{Float64,2}, datf)
        Y = dat[:,end]
        Z = dat[:,1:end-1]
        dat = nothing
        n = length(Y)
    end


    X = [ones(n) Z]
    if case != 10 && case != 11
        P = 1 .- 1 ./ (1 .+ exp.(vec(X * beta0)));
        Y = [rand(Bernoulli(P[i])) for i in 1:n];
    end
    fit_f = getMLE(X, Y, ones(n))
    # beta_f = fit_f[1]

    # F = svd(X)
    # beta_f = F.V * Diagonal(1 ./ F.S) * F.U' * Y
    # lv = sqrt.(vec(sum(X.^2, dims=2)))
    # # e = Y - X * beta_f
    # # lv = abs.(e) .* sqrt.(vec(sum(X.^2, dims=2)))
    # # lv = vec(sum(F.U.^2, dims=2))
    # # lv = lv ./ sum(lv)

    return X, Y, fit_f[1]
end
