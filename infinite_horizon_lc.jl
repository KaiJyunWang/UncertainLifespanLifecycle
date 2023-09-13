using Interpolations, Distributions, BenchmarkTools, Plots, NLsolve
using LinearAlgebra, Parameters, Random, LaTeXStrings, Expectations
using CUDA, CUDAKernels, KernelAbstractions, Tullio, LaTeXStrings 
using Optim: maximize, maximum, maximizer


#infinite horizon


function updated_dying_prob(prior, Tm_1, Tm_2)
    dp = zeros(Tm_1+1, Tm_1+1)
    #candidates of dying pmf 
    f_1(x, t) = (Tm_1-t+1≥1 ? pdf(DiscreteUniform(1, Tm_1-t+1), x) : 0.0)
    f_2(x, t) = (Tm_2-t+1≥1 ? pdf(DiscreteUniform(1, Tm_2-t+1), x) : 0.0)
    for t in 1:Tm_1+1
        @tullio dp[x, $t] = $prior*f_1(x, $t) + (1-$prior)*f_2(x, $t)
        prior = prior*(1-f_1(1, t))/(prior*(1-f_1(1, t))+(1-prior)*(f_2(1, t)))
    end
    return dp
end


function T!(v;para, tol = 1e-10)
    (;γ, y, ξ, r, α, β, b, κ, dp, Tm_1, t, μ, s) = para
    
    #interest rate
    ret = 1+r
    #consumption utility
    u(c) = (c>0 ? (γ!=1 ? c^(1-γ)/(1-γ) : log(c)) : -100.0)
    #bequest utility
    φ(b) = (b>-κ ? (γ!=1 ? α*(1+b/κ)^(1-γ)/(1-γ) : α*log(1+b/κ)) : -100.0)
    y_extend = reduce(vcat, ([minimum(ξ)*y[1]], y, [maximum(ξ)*y[length(y)]]))
    #v should match y_extend. But how?
    v_func = linear_interpolation((y_extend, b, t), v)
    t_vec = collect(t)

    σ = similar(v)

    #NPSC
    shocks = zeros(Tm_1+1)
    for t in 1:Tm_1+1
        shocks[t] = sum([(exp(μ+s^2/2)/ret)^x for x in 1:Tm_1+1] .* dp[:, t])
    end
    @tullio NPSC[i, j, k, t] := y[i+0]/(1-exp(μ+s^2/2)/$ret)*(1-shocks[t])+$ret*b[j]+b[k]
    

    #Optim approach
    #=
    for (i, y_val) in enumerate(y)
        for (j, b_val) in enumerate(b)
            for t in t_vec[1:length(t_vec)-1]
                res = maximize(a -> u(y_val + ret * b_val - a) + β * dp[t] * φ(ret * a) + β * (1-dp[t]) * mean(v_func.(y_val.*ξ, a, t+1)), tol, y_val)
                Tv[i+1, j, t] = maximum(res)
                σ[i+1, j, t] = maximizer(res)
            end
        end
    end
    =#

    #tullio-findmax approach
    
    
    @tullio Tv_candi[i, j, k, t] := (NPSC[i+0, j+0, k+0, t+0]≥0.0 ? u(y[i+0] + $ret * b[j+0] - b[k+0]) + $β * dp[1, t+0] * φ($ret * b[k+0]) + $β * (1-dp[1, t+0]) * mean(v_func.(y[i+0]*ξ, b[k+0], t+1)) : -100.0) (t in 1:Tm_1)
    for i in 1:length(y)
        for j in 1:length(b)
            for t in t_vec[1:length(t_vec)-1]
                (v[i+1, j, t], ind) = findmax(Tv_candi[i, j, :, t])
                σ[i+1, j, t] = b[ind]
            end
        end
    end
    
    return (v = v, σ = σ)
end

function infinite_horizon_lc(initial_v;ihlc, iter = 1000, m = 3, show_trace = false)
    (;γ, y, μ, s, ξ, r, α, β, b, κ, dp, Tm_1, t) = ihlc
    para = @with_kw (γ = γ, y = y, ξ = ξ, r = r, α = α, β = β, b = b, κ = κ, dp = dp, Tm_1 = Tm_1, t = t, μ = μ, s = s)
    para = para()
    results = fixedpoint(v -> T!(v;para).v, initial_v; iterations = iter, m, show_trace) # Anderson iteration # somehow not able to use all memory
    return (;v_star = results.zero, σ = T!(results.zero;para).σ, results)
end



prior, Tm_1, Tm_2 = 1e-30, 50, 30
dp = updated_dying_prob(prior, Tm_1, Tm_2)
plot(1:50, dp[1, 1:50])

immortal = zeros(size(dp))

IHLC = @with_kw (γ = 1.0, y = range(1e-5, 10.0, length = 20), μ = 0.0, s = 0.1, 
ξ = exp.(μ .+ s * randn(250)), r = 0.02, α = 0.0, β = 1/(1+r), b = range(-25.0, 25.0, length = 30), 
κ = 1000.0, dp = immortal, Tm_1 = 50, t = 1:Tm_1+1)
ihlc = IHLC()

para = @with_kw (γ = ihlc.γ, y = ihlc.y, ξ = ihlc.ξ, r = ihlc.r, α = ihlc.α, β = ihlc.β, b = ihlc.b, κ = ihlc.κ, dp = ihlc.dp, Tm_1 = ihlc.Tm_1, t = ihlc.t, μ = ihlc.μ, s = ihlc.s)
para = para()


v = ones(length(ihlc.y)+2, length(ihlc.b), length(ihlc.t))

plot(ihlc.y, v[2:end-1, 15, 1], label = "initial v")
for i in 1:5
    v = T!(v;para).v
    plot!(ihlc.y, v[2:end-1, 15, 1], label = "")
end
plot!(ihlc.y, T!(v;para).v[2:end-1, 15, 1], label = "")

v_star, policy, res = infinite_horizon_lc(v;ihlc)

ξ = exp.(ihlc.μ .+ ihlc.s * randn(49))
y = zeros(50)
y[1] = 4.0
for t in 1:49
    y[1+t] = y[t]*ξ[t]
end
plot(1:50, y, label = "y")

c = zeros(50)
b = zeros(51)
b[1] = 0.0
for t in 1:50
    policy_func = LinearInterpolation((ihlc.y, ihlc.b), policy[2:1+length(ihlc.y), :, t])
    b[t+1] = policy_func(y[t], b[t])
    c[t] = y[t]+(1+ihlc.r)*b[t]-b[t+1] 
end

plot!(1:50, b[2:51], label = "b")
plot!(1:50, c, label = "c")



plot(ihlc.y, v_star[2:1+length(ihlc.y), 15, 1], label = L"v(y, b = 0.0, t = 1)", xlabel = "y")
for t in 2:50
    plot!(ihlc.y, v_star[2:1+length(ihlc.y), 15, t], label = L"v(y, b = 0.0, t = %$t)")
end
plot!(ihlc.y, zeros(length(ihlc.y)))
