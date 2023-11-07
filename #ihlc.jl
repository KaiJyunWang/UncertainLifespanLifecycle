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

gm0 = gm([0.002,0.075,0.001])
gm1 = gm([0.001,0.07,0.001])
H_0 = MarkovChain([0.7 0.3;0.4 0.6])
H_1 = MarkovChain([0.3 0.7;0.1 0.9])


#simulation parameters
N = 10000
periods = 10
#type percentage (real prior for being helthy)
s = 0.3
type = zeros(Bool, N)
ages = zeros(Int64, N, periods)
die = zeros(Bool, N, periods)
health = zeros(Union{Missing, Int64}, N, periods)

s = fill(s, length(gm1.dp))
gm0_ex = vcat(gm0.dp,zeros(length(gm1.dp)-length(gm0.dp)))
for a in 2:length(s)
    s[a] = s[a-1]*(1-gm1.dp[a-1])/(s[a-1]*(1-gm1.dp[a-1])+(1-s[a-1])*(1-gm0_ex[a-1]))
end
#assume that the age dist. is DiscreteUniform -- does not really matter.
@tullio ages[i,1] = rand(DiscreteUniform(65, $gm1.ceiling))
#0:unhealthy 1:healthy
@tullio type[i] = rand(Bernoulli(s[ages[i,1]]))
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
for i in 1:N
    health[i,:] = (type[i] == 0 ? simulate(H_0, periods) : simulate(H_1, periods))
end
for j in 2:periods
    for i in 1:N
        health[i,j] = (sum(die[i,1:j-1]) ==0 ? health[i,j] : missing)
    end
end 

df = DataFrame(id = vec(repeat(transpose(collect(1:N)),periods)), 
                time = repeat(collect(1:10),N), age = vec(transpose(ages)),
                die = vec(transpose(die)), health = vec(transpose(health)))

#μ0 = x[1]*exp(x[2]*t)+x[3]; μ1 = x[4]*exp(x[5]*t)+x[6]
#H_1 = [x[7], x[8];1-x[7], 1-x[8]]; ht2 = [x[9], x[10];1-x[9]x[10],1-x[10]]
#s1 = x[11] :type healthy
df_h = filter(row -> !ismissing(row.health), df)

df_h[df_h.time .!= 10, :health]
pre_h = zeros(Union{Missing, Int64}, nrow(df_h))
for i in 1:nrow(df_h) 
    pre_h[i] = (df_h.time[i] != 1 ? df_h[i-1,:health] : missing)
end
df_h.pre_health = pre_h
ldf_h = dropmissing(df_h)
#=
function simulator_fs(x)
    #parameters
    gm0 = gm([x[1],x[2],x[3]])
    gm1 = gm([x[4],x[5],x[6]])
    H_0 = MarkovChain([x[7] 1-x[7];x[8] 1-x[8]])
    H_1 = MarkovChain([x[9] 1-x[9];x[10] 1-x[10]])
    s = x[11]
    N = 10000
    type = zeros(Bool, N)
    ages = zeros(Int64, N, periods)
    d = zeros(Bool, N, periods)
    health = zeros(Union{Missing, Int64}, N, periods)

    #0:unhealthy 1:healthy
    type = rand(Bernoulli(s),N)
    #assume that the age dist. is DiscreteUniform
    @tullio ages[i,1] = (type[i] == 0 ? rand(DiscreteUniform(1, $gm0.ceiling-1)) : rand(DiscreteUniform(1, $gm1.ceiling-1)))
    for j in 2:periods
        ages[:,j] = ages[:,j-1] .+ 1
    end
    #draw whether each person dies or not
    for i in 1:N
        d[i,1] = (type[i] == 0 ? rand(Bernoulli(gm0.dp[ages[i,1]])) : rand(Bernoulli(gm1.dp[ages[i,1]])))
    end
    for j in 2:periods
        for i in 1:N
            if type[i] == 0
                d[i,j] = (sum(d[i,1:j-1]) == 0 ? rand(Bernoulli(gm0.dp[ages[i,j]])) : 0)
            else
                d[i,j] = (sum(d[i,1:j-1]) == 0 ? rand(Bernoulli(gm1.dp[ages[i,j]])) : 0)
            end
        end
    end
    #draw the health status at each periods
    for i in 1:N
        health[i,:] = (type[i] == 0 ? simulate(H_0, periods) : simulate(H_1, periods)) .- 1
    end
    for j in 2:periods
        for i in 1:N
            health[i,j] = (sum(d[i,1:j-1]) ==0 ? health[i,j] : missing)
        end
    end 

    df_sim = DataFrame(id = vec(repeat(transpose(collect(1:N)),periods)), type = vec(repeat((type),periods)),
                time = repeat(collect(1:10),N), age = vec(transpose(ages)),
                die = vec(transpose(d)), health = vec(transpose(health)))
    
    return df_sim
end 
combine(groupby(dropmissing(simulator_fs(0.1*ones(11))), :age), :type => mean).type_mean
=#
#simulated prior
#MLE in first stage
function L_fs(x)
    
    if  prod((x .> 0) .* (x .< 1))*(real(log(complex((1-x[3])/x[4])))/x[5]+1 ≥ maximum(df_h.age)) == 1
        μ0(a) = min(x[1]*exp(x[2]*a)+x[3],1.0)
        μ1(a) = min(x[4]*exp(x[5]*a)+x[3],1.0)
        H_0 = [x[6] 1-x[6] ;x[7] 1-x[7]]
        H_1 = [x[8] 1-x[8] ;x[9] 1-x[9]]
        #adjust time by ourselves
        s = fill(x[10], maximum(df_h.age))
        for a in 2:maximum(df_h.age)
            s[a] = s[a-1]*(1-μ1(a-1))/(s[a-1]*(1-μ1(a-1))+(1-s[a-1])*(1-μ0(a-1)))
        end
        @tullio H[a,l,m] := s[a]*H_1[l,m]+(1-s[a])*H_1[l,m]
        
        @tullio l_μ[i] := logpdf(Bernoulli(s[df_h.age[i]]*μ1(df_h.age[i])+(1-s[df_h.age[i]])*μ0(df_h.age[i])),df_h.die[i])
        @tullio l_h[i] := log(H[ldf_h.age[i],ldf_h.pre_health[i],ldf_h.health[i]])

        return sum(l_μ)+sum(l_h)
    else
        return -1e100*sum(abs.(x))
    end
end

plot(65:99, s[65:end])

L_fs([0.002,0.075,0.001,0.001,0.07,0.7,0.4,0.3,0.1,0.3])
res_mle_fs = optimize(x -> -L_fs(x), 0.1*ones(10)) 
println(res_mle_fs.minimizer)
L_fs(res_mle_fs.minimizer)
x̂ = res_mle_fs.minimizer
#may be the problem of likelihood functions
gm0_mle = gm(res_mle_fs.minimizer[1:3])
gm1_mle = gm([res_mle_fs.minimizer[4],res_mle_fs.minimizer[5],res_mle_fs.minimizer[3]])

ForwardDiff.gradient(L_fs,[0.002,0.075,0.001,0.001,0.07,0.7,0.4,0.3,0.1,0.3])
ForwardDiff.gradient(L_fs,res_mle_fs.minimizer)

plot(65:99, x -> min(x̂[1]*exp(x̂[2]*x)+x̂[3],1.0), label = L"̂μ_0")
plot!(65:99, x -> min(0.002*exp(0.075*x)+0.001,1.0), label = L"μ^*_0")
plot!(65:99, x -> min(x̂[4]*exp(x̂[5]*x)+x̂[3],1.0), label = L"̂μ_1")
plot!(65:99, x -> min(0.001*exp(0.07*x)+0.001,1.0), label = L"μ^*_1")
s0 = fill(x̂[10], length(gm1_mle.dp))
gm0_mle_ex = vcat(gm0_mle.dp,ones(length(gm1_mle.dp)-length(gm0_mle.dp)))
for a in 2:length(s0)
    s0[a] = s0[a-1]*(1-gm1_mle.dp[a-1])/(s0[a-1]*(1-gm1_mle.dp[a-1])+(1-s0[a-1])*(1-gm0_mle_ex[a-1]))
end
plot(65:99,s0[65:99],label = L"ŝ")
plot!(65:99,s[65:99],label = L"s")
#another approach
#try only lifetable first then H
function L1(x)

end



#model solver
gm0_aug = zeros(size(gm1.dist))
gm0_aug[1:gm0.ceiling+1, 1:gm0.ceiling+1] = gm0.dist
IHLC = @with_kw (Tm_1 = max(gm0.ceiling,gm1.ceiling), t = 1:Tm_1-65+1, γ = 1.0, 
μ_y = 5.0, s_y = 0.3, ε_y = s_y*randn(250), δ = 0.5,
y = range(μ_y-6*s_y, μ_y+6*s_y, length = 20), r = 0.02, α = 5.0, β = 1/(1+r), 
b = range(-5.0, 10.0, length = 51), κ = 25.0, dp1 = gm1.dist, dp2 = gm0_aug,
hp1 = [0.3 0.1; 0.7 0.9], hp2 = [0.7 0.4; 0.3 0.6], hprior = range(0.0, 1.0, length = 11))
ihlc = IHLC()

ihlc = @with_kw (γ = 0.7, μ_y = 5.0, s_y = 0.3, ε_y = s_y*randn(250), r = 0.02, 
                α = 5.0, β = 0.95, κ = 10.0, δ = 0.5, gm0 = gm0, gm1 = gm1, 
                H_0 = [0.7 0.3;0.4 0.6], H_1 = [0.3 0.7;0.1 0.9])
ihlc = ihlc()

function backward_solve(ihlc)
    (;γ, μ_y, s_y, ε_y, r, α, β, κ, δ, gm0, gm1, H_0, H_1) = ihlc
    life_ceil = max(gm0.ceiling, gm1.ceiling)+1
    gm0_aug = zeros(size(gm1.dist))
    gm0_aug[1:gm0.ceiling+1, 1:gm0.ceiling+1] = gm0.dist
    #state variables
    y = range(μ_y+minimum(ε_y), μ_y+maximum(ε_y), length = 21)
    b = range(-κ+1e-5, 30, 51)
    h = [1-δ,1]
    a = 65:life_ceil
    s = 0.0:0.05:1.0
    #utility functions
    u(c) = (c>0 ? (γ!=1 ? c^(1-γ)/(1-γ) : log(c)) : -1e10)
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
    H = zeros(length(s), length(h), length(h)) |> x -> CuArray(x)
    @tullio H[p,l,m] = s[p]*H_1[l,m]+(1-s[p])*H_0[l,m]
    #NPSC
    r_seq = [(1/(1+r))^t for t in 1:length(a)]
    @tullio comp[p,n] := dist[p,n,o]*(1-r_seq[o])
    NPSC = zeros(length(b), length(s), length(a)) |> x -> CuArray(x)
    @tullio NPSC[k,p,n] = (b[k]+($μ_y)/($r)*comp[p,n] ≥ 0)
    #backward
    policy = zeros(length(y), length(b), length(h), length(a), length(s))
    v = zeros(size(policy))
    ind = zeros(Int64, length(y), length(b), length(h), length(s))
    v_candi = zeros(length(y), length(b), length(b), length(h), length(s)) |> x -> CuArray(x)
    #for speed
    v1 = zeros(length(y), length(b), length(b)) |> x -> CuArray(x)
    v2 = zeros(length(y), length(b), length(b), length(h)) |> x -> CuArray(x)
    v3 = zeros(length(b), length(a), length(s)) |> x -> CuArray(x)
    v4 = zeros(length(s), length(h), length(h)) |> x -> CuArray(x)
    v5 = zeros(length(s), length(h), length(h), length(a)) |> x -> CuArray(x)
    v6 = zeros(length(b), length(h), length(h), length(s)) |> x -> CuArray(x)
    v7 = zeros(length(b), length(h), length(s)) |> x -> CuArray(x)

    @tullio v1[i,j,k] = u(y[i]+(1+r)*b[j]-b[k]) #consumption
    @tullio v2[i,j,k,l] = v1[i,j,k]*h[l] #health depreciation on consumption utility
    @tullio v3[k,n,p] = $β*dist[p,1,n]*φ(b[k]*(1+$r)) #beuest
    @tullio v4[p,l,m] = s[p]*H_1[l,m]/H[p,l,m] #health update
    @tullio v5[p,l,m,n] = min((v4[p,l,m] != 1.0 ? (v4[p,l,m] != 0.0 ? v4[p,l,m]*(1-dp1[1,n])/(v4[p,l,m]*(1-dp1[1,n])+(1-v4[p,l,m])*(1-dp0[1,n])) : 0.0) : 1.0),1.0) #death update
    
    for N in 1:length(a)-1
        n = length(a)-N
        v_func = LinearInterpolation((y,b,h,s), v[:,:,:,n+1,:])
        @tullio v6[k,l,m,p] = mean(v_func(μ_y.+ε_y,b[k],h[m],v5[p,l,m,$n]))
        @tullio v7[k,l,p] = v6[k,l,m,p]*H[p,l,m]
        @tullio v_candi[i,j,k,l,p] = v2[i,j,k,l] + v3[k,$n,p] + $β*(1-dist[p,1,$n])*v7[k,l,p]
        #findmax
        v_can = Array(v_candi)
        @tullio W[i, j, l, p] := findmax(v_can[i, j, :, l, p])
        @tullio v[i, j, l, $n, p] = W[i, j, l, p][1]
        @tullio ind[i, j, l, p] = W[i, j, l, p][2]
        @tullio policy[i, j, l, $n, p] = b[ind[i, j, l, p]]
    end 
    return (v = v, σ = policy)
end

sol = backward_solve(ihlc)

y = ihlc.μ_y .+ ihlc.s_y*randn(ihlc.gm1.ceiling-64)
y = fill(ihlc.μ_y,ihlc.gm1.ceiling-64)
bon = zeros(ihlc.gm1.ceiling-63)
con = similar(y)
h = simulate(H_0,ihlc.gm1.ceiling-63, init = 2)
s = fill(0.3,ihlc.gm1.ceiling-63)

σ_func = LinearInterpolation((range(ihlc.μ_y+minimum(ihlc.ε_y), ihlc.μ_y+maximum(ihlc.ε_y), length = 21),
                                range(-ihlc.κ+1e-5, ihlc.μ_y*(1+ihlc.r)/ihlc.r, 51), 1:2, 65:ihlc.gm1.ceiling+1,
                                0.0:0.05:1.0), sol.σ)
for t in 1:ihlc.gm1.ceiling-64
    bon[t+1] = σ_func(y[t], bon[t], h[t], 64+t, s[t])
    s[t+1] = s[t]*ihlc.H_1[h[t],h[t+1]]/(s[t]*ihlc.H_1[h[t],h[t+1]]+(1-s[t])*ihlc.H_0[h[t],h[t+1]])
    s[t+1] = (t+64 < ihlc.gm0.ceiling ? s[t+1]*(1-ihlc.gm1.dp[64+t])/(s[t+1]*(1-ihlc.gm1.dp[64+t])+(1-s[t+1])*(1-ihlc.gm0.dp[64+t])) : 1.0)
end
con = y + (1+ihlc.r)*bon[1:end-1] - bon[2:end]

plot(65:ihlc.gm1.ceiling, y, label = L"y", legend = :topleft)
plot!(65:ihlc.gm1.ceiling, bon[2:end], label = L"b")
plot!(65:ihlc.gm1.ceiling, con, label = L"c")
plot!(twinx(), 65:ihlc.gm1.ceiling, s[1:end-1], label = L"s", legend = :topright)
vspan!(vcat(64 .+ findall(x -> x != 0, h[1:end-1]-h[2:end]),ihlc.gm1.ceiling).+ 0.5, 
        color = :gray, alpha = :0.4, label = "")


#=
function backward(ihlc)
    (;Tm_1, t, γ, μ_y, s_y, ε_y, y, r, α, β, b, κ, dp1, dp2, hp1, hp2, hprior, δ) = ihlc
    ret = 1+r

    #consumption utility
    u(c) = (c>0 ? (γ!=1 ? c^(1-γ)/(1-γ) : log(c)) : -1e10)
    h = [1-δ, 1]
    #bequest utility
    φ(b) = (b>-κ ? (γ!=1 ? α*(1+b/κ)^(1-γ)/(1-γ) : α*log(1+b/κ)) : -1e10)
    #hprior updated
    #can also try simulation method
    updated_h_MC = zeros(2, 2, length(hprior)) |> x -> CuArray(x)
    @tullio updated_h_MC[m, l, n] = hprior[n]*hp1[m, l]+(1.0-hprior[n])*hp2[m, l]
    #dp
    dp = zeros(length(t), length(t), length(hprior)) |> x -> CuArray(x)
    @tullio dp[s, t, n] = dp1[s, t+65]*hprior[n] + dp2[s, t+65]*(1.0-hprior[n])
    #NPSC
    NPSC = zeros(length(b), length(t), length(hprior)) |> x -> CuArray(x)
    comp = zeros(length(t), length(hprior))
    for T in 1:length(t)-65+1
        for n in 1:length(hprior)
            comp[T, n] = sum(Array(dp[2:end, T, n]).*[1-(1/ret)^T for T in 2:length(t)])
        end
    end
    @tullio NPSC[k, t, n] = b[k] + $μ_y/$r*comp[t, n] 

    σ = zeros(length(y), length(b), 2, length(hprior), length(t))
    ind = zeros(Int64, (length(y), length(b), 2, length(hprior)))
    v = zeros(size(σ)) 
    v_candi = zeros(length(y), length(b), length(b), 2, length(hprior)) |> x -> CuArray(x)
    #for speed's sake
    v1 = zeros(length(y), length(b), length(b), length(h)) |> x -> CuArray(x)
    v2 = zeros(length(b), length(hprior)) |> x -> CuArray(x)
    v3 = zeros(length(h), length(h), length(hprior)) |> x -> CuArray(x)
    v4 = similar(v3) |> x -> CuArray(x)
    v7 = zeros(length(b), length(h), length(h), length(hprior)) |> x -> CuArray(x)
    v8 = zeros(length(b), length(h), length(hprior)) |> x -> CuArray(x)

    for t in 1:Tm_1-65+1
        T = Tm_1-65+1+1-t
        v_func = LinearInterpolation((y, b, 0.5:0.5:1, hprior), v[:, :, :, :, T+1])
        @tullio v1[i, j, k, l] = u(y[i] + $ret * b[j] - b[k])*h[l]#consumption
        @tullio v2[k, n] = $β * dp[1, $T, n] * φ($ret * b[k])#bequest
        #single prior(long=healthy)
        @tullio v3[m, l, n] = min(1.0, hp1[m, l]*hprior[n]/updated_h_MC[m, l, n])#update for health info.
        @tullio v4[m, l, n] = ((1-dp2[1, $T])*(1-v3[m, l, n]) != 0.0 ? (1-dp1[1, $T])*v3[m, l, n]/((1-dp1[1, $T])*v3[m, l, n]+(1-dp2[1, $T])*(1-v3[m, l, n])) : 1.0)
        @tullio v7[k, m, l, n] = $β * (1-dp[1, $T, n]) * mean(v_func(μ_y.+ε_y, b[k], h[m], v4[m, l, n]))
        @tullio v8[k, l, n] = v7[k, m, l, n]*updated_h_MC[m, l, n]
        @tullio v_candi[i, j, k, l, n] = (NPSC[k, $T, n] ≥ 0.0 ? v1[i, j, k, l] + v2[k, n] + v8[k, l, n] : -1e10)
        #findmax
        v_can = Array(v_candi)
        @tullio W[i, j, l, n] := findmax(v_can[i, j, :, l, n])
        @tullio v[i, j, l, n, $T] = W[i, j, l, n][1]
        @tullio ind[i, j, l, n] = W[i, j, l, n][2]
        @tullio σ[i, j, l, n, $T] = b[ind[i, j, l, n]]
    end
    return (v = v, σ = σ)
end=#

#ae^(bx)+c, a:row1. b:row2. c:row3



