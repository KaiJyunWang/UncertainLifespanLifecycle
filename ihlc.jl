#ihlc
using Interpolations, Distributions, BenchmarkTools, Plots, NLsolve
using LinearAlgebra, Parameters, Random, LaTeXStrings, Expectations
using CUDA, CUDAKernels, KernelAbstractions, Tullio
using Optim, ForwardDiff, Roots, DataFrames, KernelDensity
using MortalityTables, Distances, Statistics, QuantEcon
using Optimization, OptimizationOptimJL

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
#prob. of getting sick 
H0(a, ph) = cdf(Logistic(), -38.0+0.5*a+2.0*(ph==0))
H1(a, ph) = cdf(Logistic(), -32.0+0.4*a+0.5*(ph==0))

plot(65:99, x -> cdf(Logistic(), -32.0+0.4*x))
plot!(65:99, x -> cdf(Logistic(), -38.0+0.5*x))
plot!(65:99, x -> cdf(Logistic(), -31.5+0.4*x))
plot!(65:99, x -> cdf(Logistic(), -36.0+0.5*x))

#simulation parameters
N = 10000
periods = 10

type = zeros(Bool, N)
ages = zeros(Int64, N, periods)
die = zeros(Bool, N, periods)
health = zeros(Union{Missing, Int64}, N, periods)

#type percentage (real prior for being helthy)
s = fill(0.3, length(gm1.dp)-64)
gm0_ex = vcat(gm0.dp,ones(length(gm1.dp)-length(gm0.dp)))
for a in 2:length(s)
    s[a] = (gm0_ex[a+63] != 1.0 ? s[a-1]*(1-gm1.dp[a+63])/(s[a-1]*(1-gm1.dp[a+63])+(1-s[a-1])*(1-gm0_ex[a+63])) : 1.0)
end
#assume that the age dist. is DiscreteUniform -- does not really matter.
@tullio ages[i,1] = rand(DiscreteUniform(65, $gm1.ceiling))
#0:unhealthy 1:healthy
type = rand.(Bernoulli.(s[ages[:,1].-64]))
for j in 2:periods
    ages[:,j] = ages[:,j-1] .+ 1
end
#draw whether each person dies or not
for i in 1:N
    die[i,1] = (type[i] == 0 ? rand(Bernoulli(gm0.dp[ages[i,1]])) : rand(Bernoulli(gm1.dp[ages[i,1]])))
end
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

@tullio health[i,1] = rand(Bernoulli(0.5))
for t in 2:periods
    @tullio health[i,$t] = (type[i] == 0 ? rand(Bernoulli(1-H0(ages[i,$t], health[i,$t-1]))) : rand(Bernoulli(1-H1(ages[i,$t], health[i,$t-1]))))
end
for j in 2:periods
    for i in 1:N
        health[i,j] = (sum(die[i,1:j-1]) ==0 ? health[i,j] : missing)
    end
end 

df = DataFrame(id = vec(repeat(transpose(collect(1:N)),periods)), 
                time = repeat(collect(1:10),N), age = vec(transpose(ages)),
                die = vec(transpose(die)), health = (vec(transpose(health)) .== 1))

#μ0 = x[1]*exp(x[2]*t)+x[3]; μ1 = x[4]*exp(x[5]*t)+x[6]
#H_1 = [x[7], x[8];1-x[7], 1-x[8]]; ht2 = [x[9], x[10];1-x[9]x[10],1-x[10]]
#s1 = x[11] :type healthy
df_h = filter(row -> !ismissing(row.health), df)
dt = combine(groupby(df_h, :id), nrow, :health => mean)
health_prob = zeros(nrow(df_h))
step = 1
for i in 1:nrow(dt)
    h_prob = fill(dt.health_mean[i],dt.nrow[i])
    health_prob[step:step + dt.nrow[i]-1] = h_prob
    step = step + dt.nrow[i]
end
df_h[!,:h_prob] = health_prob
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
function L_splited(x;df)
    (; df,) = df
    μ(a) = min(x[1]*exp(x[2]*a)+x[3],1.0)
    #prob of sick
    H(a, ph) = cdf(Logistic(), x[4]+x[5]*a+x[6]*(ph==0))
    pre_h = zeros(Union{Missing, Int64}, nrow(df))
    for i in 1:nrow(df) 
        pre_h[i] = (df.time[i] != 1 ? df[i-1,:health] : missing)
    end
    df.pre_health =(pre_h .== 1)
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
x = [0.0001,0.06,0.0001,-40.0,0.2,0.3]

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
plot(gm(res_unhealthy.minimizer[1:3]).dp, label = L"̂μ_0")
plot!(gm0.dp, label = L"μ_0*")
plot!(65:(64+length(combine(groupby(unhealthy_df,:age), :die => mean).die_mean)), combine(groupby(unhealthy_df,:age), :die => mean).die_mean, label = L"μ_0")

function L_combined(θ)
    split = θ[1]
    guessed_bad_life_ceiling = θ[2]
    df_h[!,:guess_type] = (df_h.h_prob .≥ split) .| (df_h.age .> guessed_bad_life_ceiling)
    pre_h = zeros(Union{Missing, Int64}, nrow(df_h))
    for i in 1:nrow(df_h) 
        pre_h[i] = (df_h.time[i] != 1 ? df_h[i-1,:health] : missing)
    end
    df_h.pre_health =(pre_h .== 1)
    ldf_h = dropmissing(df_h)
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

res_fs = optimize(θ -> -L_combined(θ), [0.5,85])
res_fs.minimizer

#MLE in first stage
#adjust to logit
#x[1:3]: type 0 Mortality
#x[4:5]: type 1 Mortality (θ_13 = x[3])
#x[6:8]: type 0 Health transition logit
#x[9:11]: type 1 Health transition logit
#x[12]: age 65 type 1 proportion


#model solver
gm0_aug = zeros(size(gm1.dist))
gm0_aug[1:gm0.ceiling+1, 1:gm0.ceiling+1] = gm0.dist
gm0_aug[1,gm0.ceiling+1:end-1] .= 1.0

ihlc = @with_kw (γ = 0.7, μ_y = 5.0, s_y = 0.3, ε_y = s_y*randn(250), r = 0.02, 
                α = 5.0, β = 0.95, κ = 10.0, δ = 0.5, gm0 = gm0, gm1 = gm1, 
                β_H0 = [-38.0,0.5,2.0], β_H1 = [-32.0,0.4,0.5])
ihlc = ihlc()


function backward_solve(ihlc)
    (;γ, μ_y, s_y, ε_y, r, α, β, κ, δ, gm0, gm1, β_H0, β_H1) = ihlc
    life_ceil = max(gm0.ceiling, gm1.ceiling)+1
    gm0_aug = zeros(size(gm1.dist))
    gm0_aug[1:gm0.ceiling+1, 1:gm0.ceiling+1] = gm0.dist
    gm0_aug[1,gm0.ceiling+1:end-1] .= 1.0
    H0(a, ph) = cdf(Logistic(), β_H0[1] + β_H0[2]*a + β_H0[3]*(ph==1))
    H1(a, ph) = cdf(Logistic(), β_H1[1] + β_H1[2]*a + β_H1[3]*(ph==1))

    #state variables
    y = range(μ_y+minimum(ε_y), μ_y+maximum(ε_y), length = 21)
    b = range(-κ+1e-5, 30, 51)
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
npsc = Array(sol.NPSC)
findall(x -> (x ≈ 1.0), sum(sol.D, dims = 2))#dist need to be adjusted
heatmap(sol.σ[:,:,1,22,end])
heatmap(Array(sol.D)[:,:,1])
heatmap(sign.(npsc[:,:,24]))

y = ihlc.μ_y .+ ihlc.s_y*randn(ihlc.gm1.ceiling-64)
y = fill(ihlc.μ_y,ihlc.gm1.ceiling-64)
bon = zeros(ihlc.gm1.ceiling-63)
con = similar(y)
h = fill(1,length(y))
for t in 2:length(y)
    h[t] = rand(Bernoulli(1-H0(64+t,h[t-1])))
end
s = fill(0.3,ihlc.gm1.ceiling-63)

σ_func = LinearInterpolation((range(ihlc.μ_y+minimum(ihlc.ε_y), ihlc.μ_y+maximum(ihlc.ε_y), length = 21),
                                range(-ihlc.κ+1e-5, ihlc.μ_y*(1+ihlc.r)/ihlc.r, 51), 0:1, 65:ihlc.gm1.ceiling+1,
                                0.0:0.05:1.0), sol.σ)
v_func = LinearInterpolation((range(ihlc.μ_y+minimum(ihlc.ε_y), ihlc.μ_y+maximum(ihlc.ε_y), length = 21),
range(-ihlc.κ+1e-5, ihlc.μ_y*(1+ihlc.r)/ihlc.r, 51), 0:1, 65:ihlc.gm1.ceiling+1,
0.0:0.05:1.0), sol.v)
v_func(y[1],0,1,65,0.3)
v_func(y[35],-9,0,86,0.1)
for t in 1:ihlc.gm1.ceiling-64
    bon[t+1] = σ_func(y[t], bon[t], h[t], 64+t, s[t])
    s[t+1] = (h[t] == 0 ? s[t]*H1(64+t,h[t])/(s[t]*H1(64+t,h[t])+(1-s[t])*H0(64+t,h[t])) : s[t]*(1-H1(64+t,h[t]))/(s[t]*(1-H1(64+t,h[t]))+(1-s[t])*(1-H0(64+t,h[t]))))
    s[t+1] = (ihlc.gm1.dp[64+t] != 1 ? s[t+1]*(1-ihlc.gm1.dp[64+t])/(s[t+1]*(1-ihlc.gm1.dp[64+t])+(1-s[t+1])*(1-gm0_ex[64+t])) : 1.0)
end
con = y + (1+ihlc.r)*bon[1:end-1] - bon[2:end]
D_func = LinearInterpolation((0.0:0.05:1.0,1:ihlc.gm1.ceiling-63,65:ihlc.gm1.ceiling+1),sol.D)

plt = plot(65:ihlc.gm1.ceiling, y, label = L"y", legend = :topleft)
plot!(65:ihlc.gm1.ceiling, bon[2:end], label = L"b")
plot!(65:ihlc.gm1.ceiling, con, label = L"c")
plot!(twinx(), 65:ihlc.gm1.ceiling, [s[1:end-1],x -> D_func(s[x-64],1,x)], label = [L"s" L"dp"] , legend = :topright, color = [:black :brown])
vspan!(vcat(64 .+ findall(x -> x != 0, h[1:end-1]-h[2:end]),ihlc.gm1.ceiling).+ 0.5, 
        color = :gray, alpha = :0.4, label = "")

plot(h)
savefig(plt, "save_plt.png")





