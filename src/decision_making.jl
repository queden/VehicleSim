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
function create_callback_generator(; max_vel=10.0, trajectory_length=10, R = Diagonal([0.1, 0.5]), timestep=0.2)

    # Define symbolic variables for all inputs, as well as trajectory

    X¹, X², X³, r¹, r², r³, tr, lw, tc, Z = let
        @variables(X¹[1:4], X²[1:4], X³[1:4], r¹, r², r³, tr, lw, tc[1:2], Z[1:6*trajectory_length]) .|> Symbolics.scalarize
    end

    states, controls = decompose_trajectory(Z)
    all_states = [[X¹,]; states]
    # vehicle_2_prediction = constant_velocity_prediction(X², trajectory_length, timestep, tc)
    # vehicle_3_prediction = constant_velocity_prediction(X³, trajectory_length, timestep, tc)
    cost_val = sum(stage_cost(x, u, R) for (x,u) in zip(states, controls))
    cost_grad = Symbolics.gradient(cost_val, Z)

    constraints_val = Symbolics.Num[]
    constraints_lb = Float64[]
    constraints_ub = Float64[]

    for k in 1:trajectory_length

        # trajectory must obey physics
        append!(constraints_val, all_states[k+1] .- evolve_state(all_states[k], controls[k], timestep))
        append!(constraints_lb, zeros(4))
        append!(constraints_ub, zeros(4))

        # # trajectory must stay within track

        # cent_dist = (states[k][1:2].-tc)'*(states[k][1:2].-tc)

        # # lets keep it a bit away from the sides for aesthetics
        # clearance = 1

        # # drive outside min circle
        # append!(constraints_val, cent_dist - (tr - lw/2 + clearance + r¹)^2)

        # # drive inside max circle
        # append!(constraints_val, (tr + lw/2 - clearance - r¹)^2 - cent_dist)

        # # lane_width - r gives us a smaller width that the center of our vehicle must adhere to

        # append!(constraints_lb, zeros(2)) 
        # append!(constraints_ub, fill(Inf, 2)) 

        # # trajectory must stay collision-free
        # append!(constraints_val, collision_constraint(states[k], vehicle_2_prediction[k], r¹, r²))
        # append!(constraints_val, collision_constraint(states[k], vehicle_3_prediction[k], r¹, r³))
        # append!(constraints_lb, zeros(2))
        # append!(constraints_ub, fill(Inf, 2))

        # trajectory must obey velocity constraints
        append!(constraints_val, states[k][3])
        append!(constraints_lb, 0.0)
        append!(constraints_ub, max_vel)
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
        cost_fn = Symbolics.build_function(cost_val, [Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc]; expression)
        (Z, X¹, X², X³, r¹, r², r³, tr, lw, tc) -> cost_fn([Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc])
    end

    full_cost_grad_fn = let
        cost_grad_fn! = Symbolics.build_function(cost_grad, [Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc]; expression)[2]
        (grad, Z, X¹, X², X³, r¹, r², r³, tr, lw, tc) -> cost_grad_fn!(grad, [Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc])
    end

    full_constraint_fn = let
        constraint_fn! = Symbolics.build_function(constraints_val, [Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc]; expression)[2]
        (cons, Z, X¹, X², X³, r¹, r², r³, tr, lw, tc) -> constraint_fn!(cons, [Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc])
    end

    full_constraint_jac_vals_fn = let
        constraint_jac_vals_fn! = Symbolics.build_function(jac_vals, [Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc]; expression)[2]
        (vals, Z, X¹, X², X³, r¹, r², r³, tr, lw, tc) -> constraint_jac_vals_fn!(vals, [Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc])
    end
    
    full_hess_vals_fn = let
        hess_vals_fn! = Symbolics.build_function(hess_vals, [Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc; λ; cost_scaling]; expression)[2]
        (vals, Z, X¹, X², X³, r¹, r², r³, tr, lw, tc, λ, cost_scaling) -> hess_vals_fn!(vals, [Z; X¹; X²; X³; r¹; r²; r³; tr; lw; tc; λ; cost_scaling])
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

"""
The physics model used for motion planning purposes.
Returns X[k] when inputs are X[k-1] and U[k]. 
Uses a slightly different vehicle model than presented in class for technical reasons.
"""
function evolve_state(X, U, Δ)
    V = X[3] + Δ * U[1] 
    θ = X[4] + Δ * U[2]
    X + Δ * [V*cos(θ), V*sin(θ), U[1], U[2]]
end

"""
Cost at each stage of the plan
"""
function stage_cost(X, U, R)
    cost = -0.1*X[3] + U'*R*U
    return cost
end

function generate_trajectory(starting_state, callbacks; trajectory_length = 10)
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
            callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, z)
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

    Ipopt.AddIpoptIntOption(prob, "print_level", 1)
    status = Ipopt.IpoptSolve(prob)

    if status == 0
        @info "Ipopt found soln"
    else
        @warn "Problem not cleanly solved. IPOPT status is $(status)."
    end
    states, controls = decompose_trajectory(prob.x)
    (; states, controls, status)
end