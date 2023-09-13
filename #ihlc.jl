#ihlc
using Interpolations, Distributions, BenchmarkTools, Plots, NLsolve
using LinearAlgebra, Parameters, Random, LaTeXStrings, Expectations
using CUDA, CUDAKernels, KernelAbstractions, Tullio, Surrogates
using Optim
using MortalityTables, Distances, Statistics, QuantEcon
using ForwardDiff, Roots, DataFrames


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
    return (ceiling = Tm+1, dist = dist, dp = μ[1:end-1])
end

gm_new(x) = gm(x).dp
res = gm(gm_p[:, 1])
#age trials
N = 10000
ages = rand(DiscreteUniform(1, res.ceiling-1), N) #assume that the age dist. is DiscreteUniform
@tullio d_status[i] := rand(Bernoulli(res.dp[ages[i]])) #draw whether each person dies or not
A = hcat(ages, d_status)

death_rate = zeros(res.ceiling-1)
for t in 1:length(death_rate)
    death_rate[t] = mean(d_status[findall(A[:, 1].==t)])
end
#modify!
L(x; ages = ages, d_status = d_status) = ((x[1] ≥ 0.0)&&(x[2] ≥ 0.0)&&((x[3] ≥ 0.0)) ? sum(log.(min.(1.0, x[1]*exp.(x[2]*ages).+x[3])).*d_status .+ log.(max.(1e-10, 1.0 .- min.(1.0, x[1]*exp.(x[2]*ages).+x[3]))).*(1.0 .- d_status)) : -1e10)
a = 0.0:0.0001:0.002
b = 0.0:0.01:0.2
v = zeros(length(a))
for i in 1:length(a)
    v[i] = L([0.001, 0.13, a[i]])
end
plot(v)

L(gm_p[:, 1])

DL(t) = ForwardDiff.gradient(L, t)
DL(gm_p[:, 1])
sol = maximize(x -> L(x), [0.5, 0.001, 0.05], BFGS())
x̂ = Optim.maximizer(sol)
findall(d_status.==1)

plot(death_rate, label = "realized death rate")
e_dt = gm(x̂).dist[1,1:end-1]
plot!(e_dt, label = "estimated death rate")
plot!(gm(gm_p[:,1], label = "real death_rate"))

function updated_dying_prob(prior, gm_p)
    res1 = gm(gm_p[:,1])
    res2 = gm(gm_p[:,2])
    Tm = max(res1.ceiling, res2.ceiling)
    d2_extend = zeros(Tm, Tm)
    d2_extend[1:res2.ceiling, 1:res2.ceiling] = res2.dist
    dp = zeros(Tm, Tm) |> x -> CuArray(x)
    priors = fill(prior, Tm)
    #candidates of dying pmf 
    priors[2:Tm] = [((1-priors[t-1])*d2_extend[1,t-1] != 0.0 ? priors[t-1]*(1-res1.dist[1,t-1])/(priors[t-1]*(1-res1.dist[1,t-1])+(1-priors[t-1])*d2_extend[1,t-1]) : 1.0) for t in 2:Tm]
    
    @tullio dp[s, t] = priors[t]*res1.dist[s,t] + (1-priors[t])*d2_extend[s,t]
    return (dp = dp, Tm = Tm)
end

function backward(ihlc)
    (;Tm_1, t, γ, μ_y, s_ε, ε, y, r, α, β, b, κ, dp1, dp2, hp1, hp2, hprior) = ihlc
    ret = 1+r

    #consumption utility
    u(c) = (c>0 ? (γ!=1 ? c^(1-γ)/(1-γ) : log(c)) : -1e10)
    h = [0.5, 1]
    #bequest utility
    φ(b) = (b>-κ ? (γ!=1 ? α*(1+b/κ)^(1-γ)/(1-γ) : α*log(1+b/κ)) : -1e10)
    #hprior updated
    #can also try simulation method
    updated_h_MC = zeros(2, 2, length(hprior)) |> x -> CuArray(x)
    @tullio updated_h_MC[m, l, n] = hprior[n]*hp1[m, l]+(1.0-hprior[n])*hp2[m, l]
    #dp
    dp = zeros(length(t), length(t), length(hprior)) |> x -> CuArray(x)
    @tullio dp[s, t, n] = dp1[s, t]*hprior[n] + dp2[s, t]*(1.0-hprior[n])
    #NPSC
    NPSC = zeros(length(b), length(t), length(hprior)) |> x -> CuArray(x)
    comp = zeros(length(t), length(hprior))
    for T in 1:length(t)
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

    for t in 1:Tm_1
        T = Tm_1+1-t
        v_func = LinearInterpolation((y, b, 0.5:0.5:1, hprior), v[:, :, :, :, T+1])
        @tullio v1[i, j, k, l] = u(y[i] + $ret * b[j] - b[k])*h[l]#consumption
        @tullio v2[k, n] = $β * dp[1, $T, n] * φ($ret * b[k])#bequest
        #priors
        #=
        @tullio v3[m, l, n] = min(1.0, hp1[m, l]*hprior[n]/updated_h_MC[m, l, n])#health update for health info.
        @tullio v4[o] = (1-dprior[o] != 0.0 ? dprior[o]*(1.0-dp1[1,$T])/(dprior[o]*(1.0-dp1[1,$T])+(1.0-dprior[o])*(1.0-dp2[1,$T])) : 1.0) #death update for living

        @tullio v5[m, l, n, o] = v4[o] * $p1 /($p1 + $p2) + (1-v4[o])*$p2/(1-$p1-$p3) #health update for living info.
        @tullio v5[o] = (dp[1, $T, o] != 1.0 ? (1.0-dp1[1, $T])*dprior[o]/(1.0-dp[1, $T, o]) : 1.0) #death update for living
        @tullio v6[m, l, n, o] = (hprior[n] != 0.0 ? $p2*v5[o]/v4[m, l, n, o] : v5[o])#death update for health
        =#
        #single prior(long=healthy)
        @tullio v3[m, l, n] = min(1.0, hp1[m, l]*hprior[n]/updated_h_MC[m, l, n])#update for health info.
        @tullio v4[m, l, n] = ((1-dp2[1, $T])*(1-v3[m, l, n]) != 0.0 ? (1-dp1[1, $T])*v3[m, l, n]/((1-dp1[1, $T])*v3[m, l, n]+(1-dp2[1, $T])*(1-v3[m, l, n])) : 1.0)
        @tullio v7[k, m, l, n] = $β * (1-dp[1, $T, n]) * mean(v_func(μ_y.+ε, b[k], h[m], v4[m, l, n]))
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
end

function periodize(x)
    u = sort(vcat(x,x.+1))
    return u
end

function plotter(vals, y)
    avg_b = zeros(length(vals))
    bonds = [Float64[] for i in vals]
    for (i, val) in enumerate(vals)
        gm_p = [0.001 0.001;
                0.13  0.26;
                0.001 0.001]
        prior = 0.9
        dp, Tm = updated_dying_prob(prior, gm_p)
        IHLC = @with_kw (Tm_1 = Tm-1, t = 1:Tm, γ = 1.0, μ_y = 5.0, s_ε = 0.5, ε = s_ε*randn(250), 
        y = range(μ_y-6*s_ε, μ_y+6*s_ε, length = 50), r = 0.02, α = 0.1, β = 1/(1+r), 
        b = range(-25.0, 25.0, length = 51), κ = val, dp = dp)
        ihlc = IHLC()
        sol = backward(ihlc)
        bon = zeros(Tm)
        for t in 1:ihlc.Tm_1
            bon[t+1] = LinearInterpolation((ihlc.y, ihlc.b), sol.σ[:, :, t])(y[t], bon[t])
        end
        bonds[i] = bon
        normal_dp = Array(dp)
        avg_b[i] = sum(normal_dp[1:end-1, 1] .* bon[2:end])
    end
    return (avg_b = avg_b, bonds = bonds)
end
#ae^(bx)+c, a:row1. b:row2. c:row3
gm_p = [0.001 0.001;
        0.13  0.26;
        0.001 0.001]

#solve the lifespan ceilings and construct discretized gompertz-makeham Distributions
#ae^(bx)+c, a:row1. b:row2. c:row3. x = [a,b,c].prior = 0.2
dp_ord, Tm = updated_dying_prob(prior, gm_p)
normal_dp_ord = Array(dp_ord)

plt = plot()
for t in 1:40
    plot!(t:Tm-1, normal_dp_ord[1:end-t,t], color = RGBA(t/40, 0, 0, 0.8), label = "")
end


die_at_Tm = zeros(size(dp)) |> x -> CuArray(x) 
die_at_Tm[1,end-1] = 1.0

#p1 = P(long ∩ g), p2 = P(short ∩ g), p3 = P(long ∩ b)
Tm1, dp1 = gm(gm_p[:, 1])
Tm2, dp2_0 = gm(gm_p[:, 2])
Tm = max(Tm1, Tm2)
dp2 = zeros(Tm, Tm)
dp2[1:Tm2, 1:Tm2] = dp2_0
cuda_dp1 = CuArray(dp1)
cuda_dp2 = CuArray(dp2)


IHLC = @with_kw (Tm_1 = Tm-1, t = 1:Tm, γ = 1.0, μ_y = 5.0, s_ε = 0.3, ε = s_ε*randn(250), 
y = range(μ_y-6*s_ε, μ_y+6*s_ε, length = 20), r = 0.02, α = 5.0, β = 1/(1+r), 
b = range(-5.0, 10.0, length = 51), κ = 25.0, dp1 = cuda_dp1, dp2 = cuda_dp2,
hp1 = [0.3 0.1; 0.7 0.9], hp2 = [0.7 0.4; 0.3 0.6], hprior = range(0.0, 1.0, length = 11))
ihlc = IHLC()

sol = backward(ihlc)

#comparative statics
Random.seed!(123)
y = ihlc.μ_y .+ ihlc.s_ε*randn(ihlc.Tm_1)
mc = MarkovChain(transpose(ihlc.hp1))
Random.seed!(123)
hind = simulate(mc, ihlc.Tm_1+1, init = 2)
h = 1/2*hind[2:end]
uh = periodize(findall(h.==0.5))
vals = range(1.0, 10.0, length = 6)
ah = fill(1.0, ihlc.Tm_1)

imfs = plotter(vals, y)
plt1 = plot(vals, imfs.avg_b, xlabel = L"κ", label = L"μ_b")
plt2 = plot(y[1:length(imfs.bonds[1])-1], label = L"y")
for (i, val) in enumerate(vals)
    plot!(imfs.bonds[i][2:end], label = L"κ = %$val")
end

savefig(plt1, "μ-κ.png")
savefig(plt2, "κ.png")

con = zeros(ihlc.Tm_1)
bon = zeros(ihlc.Tm_1+1)
period = 1:ihlc.Tm_1
plt = plot(period, y, label = L"y", lw = 2)
hpr = 0.2
s = fill(hpr, Tm)
for i in 1:ihlc.Tm_1
    bon[i+1] = LinearInterpolation((ihlc.y, ihlc.b, 0.5:0.5:1, ihlc.hprior), sol.σ[:, :, :, :, i])(y[i], bon[i], h[i], hpr)
    con[i] = y[i]+(1+ihlc.r)*bon[i]-bon[i+1]
    hpr = hpr*ihlc.hp1[hind[i+1], hind[i]]*(1-ihlc.dp1[1, i])/(hpr*ihlc.hp1[hind[i+1], hind[i]]*(1-ihlc.dp1[1, i])+(1.0-hpr)*ihlc.hp2[hind[i+1], hind[i]]*(1-ihlc.dp2[1, i]))
    s[i+1] = hpr
end
plot!(bon[2:end], label = L"b", lw = 2)
plot!(con, label = L"c", lw = 2)
vspan!(uh.-0.5, color = :gray, alpha = 0.3, label = "")
plot!(twinx(), s[1:end-1], label = "", color = :black)
savefig(plt, "plt.png")
@tullio dp[i, t] := s[t]*ihlc.dp1[i, t] + (1-s[t])*ihlc.dp2[i,t]

plt = plot()
for t in 1:40
    plot!(t:Tm-1, dp[1:end-t,t], color = RGBA(0, 0, t/40, 0.8), label = "")
end

heatmap(sol.v[:, :, 2, 3, 1])
hprs = 0.0:0.1:1.0
avgb = zeros(length(hprs))
for (k, val) in enumerate(hprs)
    hpr = val
    s = fill(hpr, Tm)
    for i in 1:ihlc.Tm_1
        bon[i+1] = LinearInterpolation((ihlc.y, ihlc.b, 0.5:0.5:1, ihlc.hprior), sol.σ[:, :, :, :, i])(y[i], bon[i], h[i], hpr)
        con[i] = y[i]+(1+ihlc.r)*bon[i]-bon[i+1]
        hpr = hpr*ihlc.hp1[hind[i+1], hind[i]]/(hpr*ihlc.hp1[hind[i+1], hind[i]]+(1.0-hpr)*ihlc.hp2[hind[i+1], hind[i]])
        s[i+1] = hpr
    end
    avgb[k] = normal_dp[1:end-1,1] ⋅ bon[2:end]
end

plot(hprs, avgb, title = L"E_1[b_T]", lw = 2, label ="")

GompMake(x) = LinearInterpolation(1:gm(x[2:4]).ceiling, gm(x[2:4]).dist[1,:])(x[1])
x = vcat(1, gm_p[:, 1])
GompMake(x)
ForwardDiff.gradient(GompMake, x)