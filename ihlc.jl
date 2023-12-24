#ihlc
using Interpolations, Distributions, BenchmarkTools, Plots, NLsolve
using LinearAlgebra, Parameters, Random, LaTeXStrings, Expectations
using CUDA, CUDAKernels, KernelAbstractions, Tullio
using Optim, ForwardDiff, Roots, DataFrames, KernelDensity
using MortalityTables, Distances, Statistics, QuantEcon
using Optimization, OptimizationOptimJL, Tables, Colors

function gm(x)
    m = Makeham(x[1], x[2], x[3])
    Tm = ceil(Int, find_zero(t -> m(t)-1, 1.0))
    μ = zeros(Tm)
    μ = [min(1.0, x[1]*exp(x[2]*t)+x[3]) for t in 1:Tm+1]
    dist = zeros(Tm+1, Tm+1)
    for t in 1:Tm+1
        for s in 1:Tm+1-t
            dist[s,t] = prod(1.0 .- μ[t:t+s-2])*μ[t+s-1]
        end
    end  
    return (ceiling = Tm, dist = dist, dp = μ[1:end-1])
end

gm0 = gm([0.001,0.075,0.001])
gm1 = gm([0.001,0.07,0.001])
gm0_aug = zeros(size(gm1.dist))
gm0_aug[1:gm0.ceiling+1, 1:gm0.ceiling+1] = gm0.dist
gm0_aug[1,gm0.ceiling+1:end-1] .= 1.0
#prob. of getting sick 
H0(a, ph) = cdf(Logistic(), -38.0+0.5*a+2.0*(ph==0))
H1(a, ph) = cdf(Logistic(), -32.0+0.4*a+0.5*(ph==0))


#simulation parameters
N = 30000
periods = length(gm1.dp)-64

γ = 0.7
μ_y = 5.0
s_y = 0.3
r = 0.02
α = 6.0
β = 0.95
κ = 10.0
δ = 0.5
Random.seed!(123)
ε_y = s_y*randn(250)

β_H0 = [-38.0,0.5,2.0]
β_H1 = [-32.0,0.4,0.5]

ihlc = @with_kw (γ = γ, μ_y = μ_y, s_y = s_y, ε_y = ε_y, r = r, 
                α = α, β = β, κ = κ, δ = δ, gm0 = gm0, gm1 = gm1, 
                β_H0 = β_H0, β_H1 = β_H1)
ihlc = ihlc()

#model solver

function backward_solve(ihlc)
    (;γ, μ_y, s_y, ε_y, r, α, β, κ, δ, gm0, gm1, β_H0, β_H1) = ihlc
    life_ceil = max(gm0.ceiling, gm1.ceiling)+1
    gm0_aug = zeros(size(gm1.dist))
    gm0_aug[1:gm0.ceiling+1, 1:gm0.ceiling+1] = gm0.dist
    gm0_aug[1,gm0.ceiling+1:end-1] .= 1.0
    H0(a, ph) = cdf(Logistic(), β_H0[1] + β_H0[2]*a + β_H0[3]*(ph==1))
    H1(a, ph) = cdf(Logistic(), β_H1[1] + β_H1[2]*a + β_H1[3]*(ph==1))

    #state variables
    #y need to be adjust by ourselves
    y = range(3, 7, length = 21)
    b = range(-κ+1e-5, 30, 101)
    h = [1-δ,1]
    a = 65:life_ceil
    s = 0.0:0.05:1.0
    #utility functions
    u(c) = (c>0 ? (γ!=1 ? c^(1-γ)/(1-γ) : log(c)) : -1e30)
    φ(b) = (b>-κ ? (γ!=1 ? α*(1+b/κ)^(1-γ)/(1-γ) : α*log(1+b/κ)) : -1e10)
    #die dist.
    dp1 = CuArray(gm1.dist[1:length(a),65:end])
    dp0 = CuArray(gm0_aug[1:length(a),65:end])
    dist = zeros(length(s), length(a), length(a))
    for p in 1:length(s)
        dist[p,:,:] = s[p]*dp1+(1-s[p])*dp0
    end
    dist = CuArray(dist)
    #H
    H = zeros(length(s), length(h), length(h), length(a)) |> x -> CuArray(x)
    @tullio H[p,l,1,n] = s[p]*H1(a[n],l)+(1-s[p])*H0(a[n],l)
    @tullio H[p,l,2,n] = 1-H[p,l,1,n]
    #backward
    policy = zeros(length(y), length(b), length(h), length(a), length(s))
    v = zeros(size(policy)) 
    
    ind = zeros(Int64, length(y), length(b), length(h), length(s))
    v_candi = zeros(length(y), length(b), length(b), length(h), length(s)) |> x -> CuArray(x)
    #for speed
    v1 = zeros(length(y), length(b), length(b)) |> x -> CuArray(x)
    v2 = zeros(length(y), length(b), length(b), length(h)) |> x -> CuArray(x)
    v3 = zeros(length(b), length(a), length(s)) |> x -> CuArray(x)
    v4 = zeros(length(s), length(h), length(h), length(a)) |> x -> CuArray(x)
    v5 = zeros(length(s), length(h), length(h), length(a)) |> x -> CuArray(x)
    v6 = zeros(length(b), length(h), length(h), length(s)) |> x -> CuArray(x)
    v7 = zeros(length(b), length(h), length(s)) |> x -> CuArray(x)

    @tullio v1[i,j,k] = u(y[i]+(1+r)*b[j]-b[k]) #consumption
    @tullio v2[i,j,k,l] = v1[i,j,k]*h[l] #health depreciation on consumption utility
    @tullio v3[k,n,p] = $β*dist[p,1,n]*φ(b[k]*(1+$r)) #beuest
    @tullio v4[p,l,m,n] = (m == 1 ? s[p]*H1(a[n],l)/H[p,l,m,n] : s[p]*(1-H1(a[n],l))/H[p,l,m,n]) #health update
    @tullio v5[p,l,m,n] = min((dp0[1,n] != 1.0 ? v4[p,l,m,n]*(1-dp1[1,n])/(v4[p,l,m,n]*(1-dp1[1,n])+(1-v4[p,l,m,n])*(1-dp0[1,n])) : 1.0), 1.0) #death update
    
    #NPSC
    r_seq = [1-(1/(1+r))^t for t in 1:length(a)]
    @tullio comp[p,n] := dist[p,o+1,n]*r_seq[o] (o in 1:length(a)-1)
    NPSC = zeros(length(b), length(s), length(a)) |> x -> CuArray(x)
    @tullio NPSC[k,p,n] = b[k]*dist[p,1,n] + $μ_y/$r*comp[p,n]
    
    for N in 1:length(a)-1
        n = length(a)-N
        v_func = LinearInterpolation((y,b,h,s), v[:,:,:,n+1,:])
        @tullio v6[k,l,m,p] = mean(v_func(μ_y.+ε_y,b[k],h[m],v5[p,l,m,$n]))
        @tullio v7[k,l,p] = v6[k,l,m,p]*H[p,l,m,$n]
        @tullio v_candi[i,j,k,l,p] = (NPSC[k,p,$n] ≥ 0.0 ? v2[i,j,k,l] + v3[k,$n,p] + $β*(1-dist[p,1,$n])*v7[k,l,p] : -1e100)
        #findmax
        v_can = Array(v_candi)
        @tullio W[i, j, l, p] := findmax(v_can[i, j, :, l, p])
        @tullio v[i, j, l, $n, p] = W[i, j, l, p][1]
        @tullio ind[i, j, l, p] = W[i, j, l, p][2]
        @tullio policy[i, j, l, $n, p] = b[ind[i, j, l, p]]
    end 
    return (v = v, σ = policy, NPSC = NPSC, D = dist)
end

sol = backward_solve(ihlc)

σ_func = LinearInterpolation((range(3, 7, length = 21), range(-ihlc.κ+1e-5, 30, 101), 
                                0:1, 65:ihlc.gm1.ceiling+1,
                                0.0:0.05:1.0), sol.σ)

type = zeros(Bool, N)
ages = zeros(Int64, N, periods)
die = zeros(Bool, N, periods)
health = zeros(Union{Missing, Int64}, N, periods)
Random.seed!(123)
y = μ_y .+ s_y*randn(N,periods)

#type percentage (real prior for being helthy)
s = fill(0.3, length(gm1.dp))
gm0_ex = vcat(gm0.dp,ones(length(gm1.dp)-length(gm0.dp)))
for a in 2:length(s)
    s[a] = (gm0_ex[a] != 1.0 ? s[a-1]*(1-gm1.dp[a])/(s[a-1]*(1-gm1.dp[a])+(1-s[a-1])*(1-gm0_ex[a])) : 1.0)
end


#draw ages
Random.seed!(123)
@tullio ages[i,1] = 65
for j in 2:periods
    @tullio ages[i,$j] = ages[i,$j-1] + 1
end
#type: 0:unhealthy 1:healthy
Random.seed!(123)
type = rand.(Bernoulli.(s[ages[:,1]]))
#draw whether each person dies or not
Random.seed!(123)
for i in 1:N
    die[i,1] = (type[i] == 0 ? rand(Bernoulli(gm0.dp[ages[i,1]])) : rand(Bernoulli(gm1.dp[ages[i,1]])))
end
Random.seed!(123)
for j in 2:periods
    for i in 1:N
        if type[i] == 0
            die[i,j] = (sum(die[i,1:j-1]) == 0 ? rand(Bernoulli(gm0.dp[ages[i,j]])) : 0)
        else
            die[i,j] = (sum(die[i,1:j-1]) == 0 ? rand(Bernoulli(gm1.dp[ages[i,j]])) : 0)
        end
    end
end
#draw the health status at each periods
Random.seed!(123)
@tullio health[i,1] = rand(Bernoulli(0.5))
Random.seed!(123)
for t in 2:periods
    @tullio health[i,$t] = (type[i] == 0 ? rand(Bernoulli(1-H0(ages[i,$t], health[i,$t-1]))) : rand(Bernoulli(1-H1(ages[i,$t], health[i,$t-1]))))
end
for j in 2:periods
    for i in 1:N
        health[i,j] = (sum(die[i,1:j-1]) ==0 ? health[i,j] : missing)
    end
end 
#b_0
assets = zeros(N,periods+1)
Random.seed!(123)
assets_0 = 10.0 .+ 2.0*randn(N)
#individual priors
Random.seed!(123)
priors = rand(Beta(2.0,2.0), N, periods+1)
for i in 1:N
    for t in 1:periods
        assets[i,1+ t] = (!ismissing(health[i, t]) ? σ_func(y[i, t], assets[i, t], health[i, t], ages[i, t], priors[i, t]) : 0)
        if t != periods && ages[i,t] ≤ length(gm1.dp)
            priors[i,(1+t)] = (!ismissing(health[i, (1+t)]) ? (health[i, (1+t)] == 0 ? priors[i,t]*H1(ages[i,t],health[i,t])/(priors[i,t]*H1(ages[i,t],health[i,t])+(1-priors[i,t])*H0(ages[i,t],health[i,t])) : priors[i,t]*(1-H1(ages[i,t],health[i,t]))/(priors[i,t]*(1-H1(ages[i,t],health[i,t]))+(1-priors[i,t])*(1-H0(ages[i,t],health[i,t])))) : 0.5)
            priors[i,(1+t)] = (ihlc.gm1.dp[ages[i,t]] != 1 ? priors[i,(t+1)]*(1-ihlc.gm1.dp[ages[i,t]])/(priors[i,(t+1)]*(1-ihlc.gm1.dp[ages[i,t]])+(1-priors[i,(t+1)])*(1-gm0_ex[ages[i,t]])) : 1.0)
        end
    end
end
#=
for i in 1:N
    for t in 1:periods
        assets[i,1+ t] = (!ismissing(health[i, t]) ? σ_func(y[i, t], assets[i, t], health[i, t], ages[i, t], priors[i, t]) : 0)
        if t != periods && ages[i,t] < 100
            priors[i,(1+t)] = (!ismissing(health[i, (1+t)]) ? (health[i, (1+t)] == 0 ? priors[i,t]*H1(ages[i,t],health[i,t])/(priors[i,t]*H1(ages[i,t],health[i,t])+(1-priors[i,t])*H0(ages[i,t],health[i,t])) : priors[i,t]*(1-H1(ages[i,t],health[i,t]))/(priors[i,t]*(1-H1(ages[i,t],health[i,t]))+(1-priors[i,t])*(1-H0(ages[i,t],health[i,t])))) : 0.5)
            priors[i,(1+t)] = (ihlc.gm1.dp[ages[i,t]] != 1 ? priors[i,(t+1)]*(1-ihlc.gm1.dp[ages[i,t]])/(priors[i,(t+1)]*(1-ihlc.gm1.dp[ages[i,t]])+(1-priors[i,(t+1)])*(1-gm0_ex[ages[i,t]])) : 1.0)
        end
    end
end
=#

df = DataFrame(id = vec(repeat(transpose(collect(1:N)),periods)), 
                time = repeat(collect(1:periods),N), age = vec(transpose(ages)),
                die = vec(transpose(die)), health = (vec(transpose(health)) .== 1),
                income = vec(y), assets = vec(assets[:,2:end]), pre_assets = vec(assets[:,1:end-1]),
                consumption = vec(y + (1+r)*assets[:,1:end-1] - assets[:,2:end]))
dropmissing!(df)
maximum(df.age)
filter(row -> row.age == 91, df)
#simulation
vals = range(1.0, 20.0, length = 11)
ass_mean_trials = zeros(periods,length(vals))
for j in 1:length(vals)
    ihlc = @with_kw (γ = γ, μ_y = μ_y, s_y = s_y, ε_y = ε_y, r = r, 
                α = α, β = β, κ = vals[j], δ = δ, gm0 = gm0, gm1 = gm1, 
                β_H0 = β_H0, β_H1 = β_H1)
    ihlc = ihlc()
    sol = backward_solve(ihlc)
    σ_func = LinearInterpolation((range(3, 7, length = 21), range(-ihlc.κ+1e-5, 30, 101), 
                                0:1, 65:ihlc.gm1.ceiling+1,
                                0.0:0.05:1.0), sol.σ)
    assets = zeros(N,periods+1)
    assets[:,1] = assets_0
    for t in 1:periods
        for i in 1:N
            assets[i,1+t] = (!ismissing(health[i, t]) ? σ_func(y[i, t], assets[i, t], health[i, t], ages[i, t], priors[i, t]) : 0)
        end
    end
    df = DataFrame(id = vec(repeat(transpose(collect(1:N)),periods)), 
                time = repeat(collect(1:periods),N), age = vec(transpose(ages)),
                die = vec(transpose(die)), health = (vec(transpose(health)) .== 1),
                income = vec(y), assets = vec(assets[:,2:end]), pre_assets = vec(assets[:,1:end-1]),
                consumption = vec(y + (1+r)*assets[:,1:end-1] - assets[:,2:end]))
    dropmissing!(df)
    gd = groupby(df, :age)
    for t in 1:length(gd)
        ass_mean_trials[t,j] = mean(gd[t].assets)
    end
end
plt = plot()
for i in 1:length(vals)
    plot!(65:91, ass_mean_trials[1:91-65+1,i], color = RGBA(i/length(vals),0,1-i/length(vals),0.8), linewidth = 2.0, label = "")
end
title!("Asset")
xlabel!("age")
savefig(plt, "kappa_plt.png")
#μ0 = x[1]*exp(x[2]*t)+x[3]; μ1 = x[4]*exp(x[5]*t)+x[6]
#H_1 = [x[7], x[8];1-x[7], 1-x[8]]; ht2 = [x[9], x[10];1-x[9]x[10],1-x[10]]
#s1 = x[11] :type healthy
df_h = filter(row -> !ismissing(row.health), df)
#view the data
vscodedisplay(df_h)
dt = combine(groupby(df_h, :id), nrow, :health => mean, :age => minimum)
health_prob = zeros(nrow(df_h))
age_minimum = similar(health_prob)
step = 1
for i in 1:nrow(dt)
    health_prob[step:step + dt.nrow[i]-1] = fill(dt.health_mean[i],dt.nrow[i])
    age_minimum[step:step + dt.nrow[i]-1] = fill(dt.age_minimum[i],dt.nrow[i])
    step = step + dt.nrow[i]
end
df_h[!,:h_prob] = health_prob
df_h[!,:first_obs_age] = convert.(Int, age_minimum)
gd = groupby(df, :age)
gd[end]
mean_assets = zeros(length(gd))
median_assets = zeros(size(mean_assets))
ass_lb = zeros(size(mean_assets))
ass_ub = zeros(size(mean_assets))
ass_min = zeros(size(mean_assets))
ass_max = zeros(size(mean_assets))
ass_var = zeros(size(mean_assets))
for t in 1:length(gd)
    median_assets[t] = quantile(gd[t].assets, 0.5)
    mean_assets[t] = mean(gd[t].assets)
    ass_lb[t] = quantile(gd[t].assets, 0.025)
    ass_ub[t] = quantile(gd[t].assets, 0.975)
    ass_min[t] = minimum(gd[t].assets)
    ass_max[t] = maximum(gd[t].assets)
    ass_var[t] = var(gd[t].assets)
end
plt = plot(65:64+length(gd), mean_assets, color = :blue, label = "mean")
plot!(65:64+length(gd), ass_ub, label = "", color = :black, linestyle = :dash)
plot!(65:64+length(gd), ass_lb, label = "", color = :black, linestyle = :dash, fillrange = ass_ub, fillalpha = 0.2)
title!("Assets")
xlabel!("Age")
plt = plot(65:64+length(gd), x -> nrow(gd[x-64]), label = "obs.")
xlabel!("Age")
savefig(plt, "demographics.png")
plt = plot(65:64+length(gd), x -> 1-mean(gd[x-64].health), label = "Sick prob.")
savefig(plt, "sick_prob.png")
#should depends on age and health state
#split on age first, then health prob
split = res_fs.minimizer[1]
guessed_bad_life_ceiling = res_fs.minimizer[2]
df_h[!,:guess_type] = (df_h.h_prob .≥ split) .| (df_h.age .> guessed_bad_life_ceiling)
pre_h = zeros(Union{Missing, Int64}, nrow(df_h))
for i in 1:nrow(df_h) 
    pre_h[i] = (df_h.time[i] != 1 ? df_h[i-1,:health] : missing)
end
df_h.pre_health =(pre_h .== 1)
ldf_h = dropmissing(df_h)

#initial point should be chosen s.t. the predicted life ceiling is greater. 
#can possibly separate H and μ
function L_splited(x;df)
    (; df,) = df
    μ(a) = min(x[1]*exp(x[2]*a)+x[3],1.0)
    #prob of sick
    H(a, ph) = cdf(Logistic(), x[4]+x[5]*a+x[6]*(ph==0))
    pre_h = zeros(Union{Missing, Int64}, nrow(df))
    for i in 1:nrow(df) 
        pre_h[i] = ((df.time[i] != 1 || i > 1) ? df.health[i-1] : missing)
    end
    df.pre_health = (pre_h .== 1)
    ldf = dropmissing(df)
    
    @tullio l_μ[i] := (μ(df.age[i])>0.0 ? log(μ(df.age[i]))*df.die[i] + log(1-μ(df.age[i]))*(1 - df.die[i]) : -1e20)
    @tullio l_H[i] := log(H(ldf.age[i],ldf.pre_health[i]))*(1-ldf.health[i]) + log(1-H(ldf.age[i],ldf.pre_health[i]))* ldf.health[i]
    return sum(l_μ)+sum(l_H)
end
healthy_df = filter(row -> row.guess_type == 1, df_h)
df = @with_kw (df = healthy_df,)
h_df = df()
unhealthy_df = filter(row -> row.guess_type == 0, df_h)
udf = @with_kw (df = unhealthy_df,)
uh_df = udf()
x = [0.0001,0.05,0.0001,-40.0,0.2,0.3]

L_splited(x;df = h_df)
res_healthy = optimize(x -> -L_splited(x; df = h_df), x)
L_splited(res_healthy.minimizer;df = h_df)
res_healthy.minimizer
plt = plot(gm(res_healthy.minimizer[1:3]).dp, label = L"̂μ_1")
plot!(gm1.dp, label = L"μ_1*")
plot!(65:99, combine(groupby(healthy_df,:age), :die => mean).die_mean, label = L"μ_1")
L_splited(x;df = uh_df)
res_unhealthy = optimize(x -> -L_splited(x; df = uh_df), x)
L_splited(res_unhealthy.minimizer;df = h_df)
res_unhealthy.minimizer
plt = plot(gm(res_unhealthy.minimizer[1:3]).dp, label = L"̂μ_0")
plot!(gm0.dp, label = L"μ_0*")
plot!(65:(64+length(combine(groupby(unhealthy_df,:age), :die => mean).die_mean)), combine(groupby(unhealthy_df,:age), :die => mean).die_mean, label = L"μ_0")

plt = plot(65:99, a -> cdf(Logistic(), res_unhealthy.minimizer[4]+res_unhealthy.minimizer[5]*a), label = L"H_0")
plot!(65:99, a -> H0(a, 1), label = L"H_0*")
title!(L"h_{t-1} = 1")

plot!(65:99, a -> cdf(Logistic(), res_healthy.minimizer[4]+res_healthy.minimizer[5]*a), label = L"H_1")
plot!(65:99, a -> H1(a, 1), label = L"H_1*")
title!(L"h_{t-1} = 1")

savefig(plt, "H.png")
function L_combined(θ)
    split = θ[1]
    guessed_bad_life_ceiling = θ[end]
    df_h[!,:guess_type] = (df_h.h_prob .≥ split) .| (df_h.age .> guessed_bad_life_ceiling)
    healthy_df = filter(row -> row.guess_type == 1, df_h)
    df = @with_kw (df = healthy_df,)
    h_df = df()
    unhealthy_df = filter(row -> row.guess_type == 0, df_h)
    udf = @with_kw (df = unhealthy_df,)
    uh_df = udf()

    res_healthy = optimize(x -> -L_splited(x; df = h_df), [0.0001,0.06,0.0001,-40.0,0.2,0.3])
    res_unhealthy = optimize(x -> -L_splited(x; df = uh_df), [0.0001,0.06,0.0001,-40.0,0.2,0.3])
    return L_splited(res_healthy.minimizer; df = h_df)+L_splited(res_unhealthy.minimizer; df = uh_df)
end 

init_θ = [0.5,95]
res_fs = optimize(θ -> -L_combined(θ), init_θ)
res_fs.minimizer


#SMM target moments
#E(b|h),E(b|a)
function SMM(θ)
    para_SMM = @with_kw (γ = θ[1], μ_y = μ_y, s_y = s_y, ε_y = ε_y, r = r, 
                        α = θ[2], β = β, κ = θ[3], δ = θ[4], gm0 = gm0, gm1 = gm1, 
                        β_H0 = β_H0, β_H1 = β_H1)
    para = para_SMM()

    sol = backward_solve(para)

    σ_func = LinearInterpolation((range(3, 7, length = 21), range(-ihlc.κ+1e-5, 30, 51), 
                        0:1, 65:ihlc.gm1.ceiling+1, 0.0:0.05:1.0), sol.σ)
    
end

#plotter
bon = zeros(ihlc.gm1.ceiling-63)
h = ones(size(bon))
y = (μ_y .+ s_y*randn(35))
s = fill(0.3,size(h))
for t in 1:ihlc.gm1.ceiling-64
    bon[t+1] = σ_func(y[t], bon[t], h[t], 64+t, s[t])
    h[t+1] = rand(Bernoulli(1-H0(64+t, h[t])))
    s[t+1] = (h[t+1] == 0 ? s[t]*H1(64+t,h[t])/(s[t]*H1(64+t,h[t])+(1-s[t])*H0(64+t,h[t])) : s[t]*(1-H1(64+t,h[t]))/(s[t]*(1-H1(64+t,h[t]))+(1-s[t])*(1-H0(64+t,h[t]))))
    s[t+1] = (ihlc.gm1.dp[64+t] != 1 ? s[t+1]*(1-ihlc.gm1.dp[64+t])/(s[t+1]*(1-ihlc.gm1.dp[64+t])+(1-s[t+1])*(1-gm0_ex[64+t])) : 1.0)
end
con = y + (1+ihlc.r)*bon[1:end-1] - bon[2:end]



plt = plot(65:ihlc.gm1.ceiling, y, label = L"y", legend = :topleft)
plot!(65:ihlc.gm1.ceiling, bon[2:end], label = L"b")
plot!(65:ihlc.gm1.ceiling, con, label = L"c")
plot!(twinx(), 65:ihlc.gm1.ceiling, s[1:end-1], label = L"s", legend = :topright, color = :black)
vspan!(vcat(64 .+ findall(x -> x != 0, h[1:end-1]-h[2:end]),ihlc.gm1.ceiling).+ 0.5, 
        color = :gray, alpha = :0.4, label = "")

savefig(plt, "plt.png")



