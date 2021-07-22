cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Plots
include("model.jl")
include("parameter.jl")

u,r = sys_solve(para, brownian)


x = [v[3] for v in r]
y = [v[4] for v in r]

plt1 = plot(x,y)
plt2 = heatmap(u[100])