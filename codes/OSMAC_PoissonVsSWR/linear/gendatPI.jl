gendat = function(n::Int, case::Int=1, beta0::Vector=ones(10))
    d = length(beta0)
    ds = d - 1
    corr  = 0.5
    sigmax = [corr+(1-corr)*(i==j) for i in 1:ds, j in 1:ds]
    if case == 1 # Normal
        Z = collect(rand(MvNormal(zeros(ds), sigmax), n)')
    else # T
        df = [5, 4, 3, 2, 1][case]
        Z = rand(MvNormal(zeros(ds), sigmax), n);
        Z = collect(Z') ./ sqrt.(rand(Chisq(df), n)./df)
    end
    Y = beta0[1] .+ Z * beta0[2:end] + randn(n)
    X = [ones(n) Z]

    F = svd(X)
    beta_f = F.V * Diagonal(1 ./ F.S) * F.U' * Y
    lv = sqrt.(vec(sum(X.^2, dims=2)))
    e = Y - X * beta_f
    # lv = abs.(e) .* sqrt.(vec(sum(X.^2, dims=2)))
    # lv = vec(sum(F.U.^2, dims=2))
    # lv = lv ./ sum(lv)
    return X, Y, beta_f, lv, e
end
