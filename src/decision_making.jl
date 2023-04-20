using Ipopt
using Symbolics
using LazySets
using Polyhedra
using LinearAlgebra
using SparseArrays
using Infiltrator
using StaticArrays

include("map.jl")

function run1()
    expression = Val{false}
    
    # states, controls = decompose_trajectory(Z)
    # cost_val = sum(stage_cost(x, u) for (x,u) in zip(states, controls))

    map = training_map()

    output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    z = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    X¹, y, Z = let
        ( @variables(X¹[1:4], y[1:2], Z[1:2*trajectory_length]) ) .|> Symbolics.scalarize
    end

    cost_val = sum(Z[i] * Z[i] for i in 1:2*trajectory_length)

    cost_grad = Symbolics.gradient(cost_val, Z)

    # @info "cost val is $cost_val"
    # @info "cost grad is $cost_grad len z is $(length(Z))"

    res = [y[1], y[2]]
    func = Symbolics.build_function(res, [y;]; expression)[1]
    func2 = Symbolics.build_function(cost_grad, [Z;]; expression)[1]

    

    @infiltrate

    output = cost_grad_fn!(grad, z)

    # output = full_cost_grad_fn(grad, z)

    @info "output is $output"

end

function run_stuff()

    map = training_map()

    callbacks = create_callback_generator(max_vel=5.0)

    # seg = get_segments(map, pos)

    # angle = latest_gt.orientation
    # yaw = QuaternionToYaw(angle)

    #trajectory = generate_trajectory(ego, V2, V3, track_radius, lane_width, track_center, callbacks, traj_length, timestep)

    # velo = latest_gt.velocity

    # @info "Velocity is type $(typeof(velo)) and value $velo"

    # TODO: velo[1] is totally wrong 
    # state = [pos[1], pos[2], yaw, velo[1], angular velo, current_segment_id]

    @info "init state"

    

    state = [0, 0, 0, 0, 0, 32]

    # get middle pos of segment
    starting = [0, 0]
    divisor = 0
    for lb in map.all_segs[32].lane_boundaries
        starting += lb.pt_a[1:2]
        starting += lb.pt_b[1:2]
        divisor += 2
    end

    starting /= divisor

    @info "Generating trajectory starting at $starting"

    trajectory = generate_trajectory(state, callbacks)

end

"""
Create functions which accepts X¹, X², X³, r¹, r², r³, a¹, b¹, a², b², as input, and each return
one of the 5 callbacks which constitute an IPOPT problem: 
1. eval_f
2. eval_g
3. eval_grad_f
4. eval_jac_g
5. eval_hess_lag

Xⁱ is the vehicle_state of vehicle i at the start of the trajectory (t=0)
rⁱ is the radius of the i-th vehicle.

The purpose of this function is to construct functions which can quickly turn 
updated world information into planning problems that IPOPT can solve.
"""
# trajectory_length=40, timestep=0.2, R = Diagonal([0.1, 0.5]), 
function create_callback_generator(; map=training_map(), max_vel=10.0, trajectory_length=15, R = Diagonal([0.1, 0.5]), timestep=0.2)

    # Define symbolic variables for all inputs, as well as trajectory
   
    # State = [x, y, velocity, yaw(steering_angle)]
    # inputs = [target velo, target steering angle]

    # define variable that has to be an integer
    # @variables x::Int y::Int

    X¹, Z = let
        ( @variables(X¹[1:6], Z[1:8*trajectory_length]) ) .|> Symbolics.scalarize
    end

    @info "Variable init X is $X¹ and Z is $Z"

    states, controls = decompose_trajectory(Z)
    
    all_states = [[X¹,]; states]
    # vehicle_2_prediction = constant_velocity_prediction(X², trajectory_length, timestep, tc)
    # vehicle_3_prediction = constant_velocity_prediction(X³, trajectory_length, timestep, tc)
   
    cost_val = sum(stage_cost(x, u) for (x,u) in zip(states, controls))

    cost_grad = Symbolics.gradient(cost_val, Z)

    constraints_val = Symbolics.Num[]
    constraints_lb = Float64[]
    constraints_ub = Float64[]

    for k in 1:trajectory_length
        # trajectory must obey physics
        append!(constraints_val, all_states[k+1] .- evolve_state(map, all_states[k], controls[k], timestep))
        append!(constraints_lb, zeros(5)) # fuck it maybe only mostly follow physics?
        append!(constraints_ub, zeros(5))

        # lane boundaries... stay in parent or a child's
        append!(constraints_val, all_states[k+1][5] - 0.5)

        # trajectory must obey velocity constraints
        # maybe being near center line is actually a score function sorta thing?
        # also is it better if conditions are differentiable? 

        curr_seg_id = all_states[k+1][5]

        pos = all_states[k+1][1:2]

        withinLaneBoundaries = Int(inside_segment_or_child(pos, curr_seg_id)) # 1 if true, 0 if false

        append!(constraints_val, withinLaneBoundaries)
        append!(constraints_lb, 1)
        append!(constraints_ub, 1) # max velo
        
        append!(constraints_val, controls[k][1])
        append!(constraints_lb, 0)
        append!(constraints_ub, 5.0) # max velo

        append!(constraints_val, controls[k][2])
        append!(constraints_lb, 0)
        append!(constraints_ub, 0.5) # max steering angle
    end

    constraints_jac = Symbolics.sparsejacobian(constraints_val, Z)
    (jac_rows, jac_cols, jac_vals) = findnz(constraints_jac)
    num_constraints = length(constraints_val)

    λ, cost_scaling = let
        @variables(λ[1:num_constraints], cost_scaling) .|> Symbolics.scalarize
    end

    lag = (cost_scaling * cost_val + λ' * constraints_val)
    lag_grad = Symbolics.gradient(lag, Z)
    lag_hess = Symbolics.sparsejacobian(lag_grad, Z)
    (hess_rows, hess_cols, hess_vals) = findnz(lag_hess)
    
    expression = Val{false}

    full_cost_fn = let
        cost_fn = Symbolics.build_function(cost_val, [Z;]; expression)
        (Z) -> cost_fn([Z;])
    end

    full_cost_grad_fn = let
        cost_grad_fn! = Symbolics.build_function(cost_grad, [Z;]; expression)[2]

        (grad, Z) -> begin
            cost_grad_fn!(grad, [Z;])
        end
    end

    full_constraint_fn = let
        constraint_fn! = Symbolics.build_function(constraints_val, [Z;]; expression)[2]
        (cons, Z) -> constraint_fn!(cons, [Z;])
    end

    full_constraint_jac_vals_fn = let
        constraint_jac_vals_fn! = Symbolics.build_function(jac_vals, [Z;]; expression)[2]
        (vals, Z) -> constraint_jac_vals_fn!(vals, [Z;])
    end
    
    full_hess_vals_fn = let
        hess_vals_fn! = Symbolics.build_function(hess_vals, [Z;cost_scaling;λ]; expression)[2]
        (vals, Z, cost_scaling, λ) -> hess_vals_fn!(vals, [Z;cost_scaling;λ])
    end

    full_constraint_jac_triplet = (; jac_rows, jac_cols, full_constraint_jac_vals_fn)
    full_lag_hess_triplet = (; hess_rows, hess_cols, full_hess_vals_fn)

    return (; full_cost_fn, 
            full_cost_grad_fn, 
            full_constraint_fn, 
            full_constraint_jac_triplet, 
            full_lag_hess_triplet,
            constraints_lb,
            constraints_ub)
end

"""
Assume z = [U[1];...;U[K];X[1];...;X[K]]
Return states = [X[1], X[2],..., X[K]], controls = [U[1],...,U[K]]
where K = trajectory_length
"""
function decompose_trajectory(z)
    K = Int(length(z) / 6)
    controls = [@view(z[(k-1)*2+1:k*2]) for k = 1:K]
    states = [@view(z[2K+(k-1)*4+1:2K+k*4]) for k = 1:K]
    return states, controls
end

function compose_trajectory(states, controls)
    K = length(states)
    z = [reduce(vcat, controls); reduce(vcat, states)]
end

"""
The physics model used for motion planning purposes.
Returns X[k] when inputs are X[k-1] and U[k]. 
Uses a slightly different vehicle model than presented in class for technical reasons.
"""
function evolve_state(map, X, U, Δ)

    # X is state [x, y, yaw, velocity, angular velocity, segment]
    # U is controls [target velocity, target steering angle]

    
    # for now assume instant changing

    V = U[1] 
    θ = X[3] + U[2] * Δ
    next = X + Δ * [V*cos(θ), V*sin(θ), U[2] * Δ, U[1], U[2], 0]

    pos = SVector{2}(next[1], next[2])
    seg_id = next[5]

    ## TODO: Overlapping segments

    # we first check if it is in same segment. then children. then all them
    if (inside_segment(map, pos, seg_id) == false)
        
        # in diff segment

        for c in map.segments[seg_id].children
            if (inside_segment(map, pos, c) == true)
                next[5] = c.id
                return next
            end
        end

        segments = get_segments(map, [pos[1], pos[2], 0])
        if (length(segments) >= 1)
            next[5] = segments[1].id
            return next
        end

        # we lost as hell but pretend same segment as last time
    end

    return next
end

# pos1 pos2 velo angle 


"""
Cost at each stage of the plan
"""
function stage_cost(X, U)
    @info "Stage cost with X = $X and U = $U"

    # for now, higher velocity is better

    return -U[1]
end

# Don't call this function until we know where we are
function generate_trajectory(starting_state, callbacks; trajectory_length = 15)
    # X1 = ego.state
    # X2 = V2.state
    # X3 = V3.state
    # r1 = ego.r
    # r2 = V2.r
    # r3 = V3.r
   
    # TODO refine callbacks given current positions of vehicles, lane geometry,
    # etc.
    # refine callbacks with current values of parameters / problem inputs
    wrapper_f = function(z) 
        # callbacks.full_cost_fn(z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center)
        callbacks.full_cost_fn(z)
    end
    wrapper_grad_f = function(z, grad)
        # callbacks.full_cost_grad_fn(grad, z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center)

        callbacks.full_cost_grad_fn(grad, z)
    end
    wrapper_con = function(z, con)
        # callbacks.full_constraint_fn(con, z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center)

        callbacks.full_constraint_fn(con, z)
    end
    wrapper_con_jac = function(z, rows, cols, vals)
        if isnothing(vals)
            rows .= callbacks.full_constraint_jac_triplet.jac_rows
            cols .= callbacks.full_constraint_jac_triplet.jac_cols
        else
            # callbacks.full_constraint_jac_triplet.full_constraint_jac_vals_fn(vals, z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center)
            callbacks.full_constraint_jac_triplet.full_constraint_jac_vals_fn(vals, z)
        end
        nothing
    end
    wrapper_lag_hess = function(z, rows, cols, cost_scaling, λ, vals)
        if isnothing(vals)
            rows .= callbacks.full_lag_hess_triplet.hess_rows
            cols .= callbacks.full_lag_hess_triplet.hess_cols
        else
            # callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, z, X1, X2, X3, r1, r2, r3, track_radius, lane_width, track_center, λ, cost_scaling)
            callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, z, cost_scaling, λ)
        end
        nothing
    end

    n = trajectory_length*6
    m = length(callbacks.constraints_lb)
    prob = Ipopt.CreateIpoptProblem(
        n,
        fill(-Inf, n),
        fill(Inf, n),
        length(callbacks.constraints_lb),
        callbacks.constraints_lb,
        callbacks.constraints_ub,
        length(callbacks.full_constraint_jac_triplet.jac_rows),
        length(callbacks.full_lag_hess_triplet.hess_rows),
        wrapper_f,
        wrapper_con,
        wrapper_grad_f,
        wrapper_con_jac,
        wrapper_lag_hess
    )

    controls = repeat([zeros(2),], trajectory_length)
    states = repeat([starting_state,], trajectory_length)
    zinit = compose_trajectory(states, controls)
    prob.x = zinit

    @info "Solving with IPOPT"

    Ipopt.AddIpoptIntOption(prob, "print_level", 1)
    status = Ipopt.IpoptSolve(prob)

    if status == 0
        @info "Ipopt found soln " * string(prob.x)
    else
        @warn "Problem not cleanly solved. IPOPT status is $(status)."
    end
    states, controls = decompose_trajectory(prob.x)
    (; states, controls, status)
end