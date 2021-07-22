using Statistics
import Base.Threads.@threads, Base.Threads.@sync, Base.Threads.@spawn
using LinearAlgebra
using StaticArrays


function brownian(u,p,dt)
    γ, Γ = p
    @inbounds begin
        # vx, vy = u1, u2
        du1 = u[1] - u[1]*γ*dt + sqrt(Γ*dt)*randn()
        du2 = u[2] - u[2]*γ*dt + sqrt(Γ*dt)*randn()

        # x,y = u3, u4
        du3 = u[3] + du1*dt
        du4 = u[4] + du2*dt
    end

    return @SVector[du1,du2,du3,du4]
end

function periodic_BC(u,BC)
    up_b, low_b = BC
    vx,vy,x,y = u 

    if x > up_b 
        x = low_b + x - up_b
    elseif x < low_b
        x = up_b - (low_b - x) 
    end

    if y > up_b 
        y = low_b + y - up_b
    elseif x < low_b
        y = up_b - (low_b - y) 
    end
    u = vx,vy,x,y
end

function diffusion(u,p)
    β, num_x = p
    @inbounds for i in 2:num_x-1
        for j in 2:num_x-1
           @views u[i,j] = @. β*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) + u[i,j]
        end
    end
    @inbounds for j in 2:num_x-1
        @views u[1,j] = @. β*(u[2,j] + u[num_x,j] + u[1,j+1] + u[1,j-1] - 4*u[1,j]) + u[1,j]
    end
    @inbounds for j in 2:num_x-1
        @views u[num_x,j] = @. β*(u[1,j] + u[num_x-1,j] + u[num_x,j+1] + u[num_x,j-1] - 4*u[num_x,j]) + u[num_x,j]
    end
    @inbounds for i in 2:num_x-1
        @views u[i,1] = @. β*(u[i+1,1] + u[i-1,1] + u[i,2] + u[i,num_x] - 4*u[i,1]) + u[i,1]
    end
    @inbounds for i in 2:num_x-1
        @views u[i,num_x] = @. β*(u[i+1,num_x] + u[i-1,num_x] + u[i,1] + u[i,num_x-1] - 4*u[i,num_x]) + u[i,num_x]
    end
    u
end



function sys_solve(para,dynaEq)

    # para of duffusion
    L = para["size of grid"]
    dx = para["dx"]
    D = para["Diffusion const"]
    source = para["source"]

    dt = dx^2/(4*D)
    β = (D*dt)/(dx^2)

    tspan = para["time span"]
    t = tspan[1]:dt:tspan[end]
    N_step = length(t)

    # Initialze solution of the grid of u(k,i,j)
    num_x =  floor(Int, L/dx)
    u0 = zeros(num_x, num_x)
    u = [u0 for i in 1:N_step]
    u[1] = u0
    # define where is 0 in our grid
    grid0 = floor(Int, num_x/2)
    @views u0[grid0, grid0] = source

    p1 = (β, num_x)

    # para of brownian
    Temp = para["Temperture"]
    k = para["Boltzmann const"] #1.380649*10^-23  #Boltzmann constant
    m = para["mass"]
    γ = para["gamma"]


    Γ = 2*γ*Temp/m
    p = (γ, Γ)
    r0 = @SArray[0.0, 0.0, 0.0, 0.0]
    r = Vector{typeof(r0)}(undef, N_step)
    r[1] = r0

    # Boundary of brownian
    low_b = -L/2 + dx
    up_b = L/2 - dx  
    BC = (up_b, low_b)


    ### main loop
    for i in 1:N_step-1
        # diffusion
        u[i+1] =copy(diffusion(u[i],p1))

        # brownian
        r[i+1] = dynaEq(r[i],p,dt)
        # periodic BC
        r[i+1] = periodic_BC(r[i+1], BC)
        
        # updata the source position 
        u[i+1][grid0+round(Int, r[i+1][3]/dx), grid0+round(Int, r[i+1][4]/dx)] = source
    end

    return u, r 
end