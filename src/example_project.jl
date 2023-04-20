struct MyLocalizationType
    field1::Int
    field2::Float64
end

struct MyPerceptionType
    field1::Int
    field2::Float64
end

function localize(gps_channel, imu_channel, localization_state_channel)
    # Set up algorithm / initialize variables
    while true
        fresh_gps_meas = []
        while isready(gps_channel)
            meas = take!(gps_channel)
            push!(fresh_gps_meas, meas)
        end
        fresh_imu_meas = []
        while isready(imu_channel)
            meas = take!(imu_channel)
            push!(fresh_imu_meas, meas)
        end
        
        # process measurements

        localization_state = MyLocalizationType(0,0.0)
        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end 
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
    # set up stuff
    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        latest_localization_state = fetch(localization_state_channel)
        
        # process bounding boxes / run ekf / do what you think is good

        perception_state = MyPerceptionType(0,0.0)
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
    end
end

using Rotations

function get_current_state(latest_gt::GroundTruthMeasurement)
    pos = latest_gt.positions

    angle = latest_gt.orientation
    yaw = QuaternionToYaw(angle)
    velo = latest_gt.velocity
 
    [pos[1], pos[2], velo[1], yaw]
end

target_map_segment = 0 # (not a valid segment, will be overwritten by message)

function decision_making(localization_state_channel, 
        perception_state_channel, 
        gt_channel, # for testing
        map, 
        target_road_segment_id, 
        socket)
    # do some setup

    # wait 5 seconds for location to be set
    @info "Waiting 5 to get accurate location info"
    sleep(1) # TODO Wait 5

    timestep=0.2
    
    @info "Creating callback gen"
    callbacks = create_callback_generator(max_vel=5.0, timestep, traj_length=15.0)

    # get initial state
    latest_gt = fetch(gt_channel)
    state = get_current_state(latest_gt)

    # get our first trajectory
    lastPath = []
    trajectory = nothing
    current_step = 1

    # Continuously generate new trajectories and save them
    errormonitor(@async while true
        # get latest state
        latest_gt = fetch(gt_channel)
        state = get_current_state(latest_gt)

        curr_seg_id = nothing
        for seg_id in lastPath
            if inside_segment(state[1:2], map.segments[seg_id])
                curr_seg_id = seg_id
                break
            end
        end

        if (curr_seg_id === nothing)

            current_segments = get_segments(map, state[1:2])

            if (length(current_segments) == 0)
                @info "Not on map, houston we are fucked"
                curr_seg_id = lastPath[1]
            else
                curr_seg_id = current_segments[1]
            end
        end

        # we are not on the path, so we need to find a new path
        @info "Not on path, finding new path"
        lastPath = find_path(curr_seg_id, target_road_segment_id)

        trajectory = generate_trajectory(state, lastPath[1:2], target_road_segment_id, callbacks)
        current_step = 1
    end
    )

    while true
        
        sleep(time_step)

        # No trajectory calculated yet
        if (trajectory === nothing) 
            continue
        end

        states, controls = decompose_trajectory(trajectory)

        if length(states) >= current_step
            # apply the current step controls
            target_angle = controls[current_step][2]
            target_velo = states[current_step][3]

            cmd = VehicleCommand(target_angle, target_velo, true)
            serialize(socket, cmd)
        end

        current_step++
    end

end

# this is prob wrong but we rolling with it for now
function QuaternionToYaw(q::SVector{4, Float64} = SVector(1.0, 0.0, 0.0, 0.0))
    # roll (x-axis rotation)
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = atan(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = sqrt(1 + 2 * (q.w * q.y - q.x * q.z))
    cosp = sqrt(1 - 2 * (q.w * q.y - q.x * q.z))
    pitch = 2 * atan(sinp, cosp) - π / 2

    # yaw (z-axis rotation)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = atan(siny_cosp, cosy_cosp)

    (yaw)
end


function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end

function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.training_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    localization_state_channel = Channel{GroundTruthMeasurement}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)

    # target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)


        sleep(0.001)
        local measurement_msg
        received = false
        while true

            # get ping from server
            

            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end
        !received && continue
        target_map_segment = measurement_msg.target_segment

        ego_vehicle_id = measurement_msg.vehicle_id
        for meas in measurement_msg.measurements

            if meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)

    # @async localize(gps_channel, imu_channel, localization_state_channel)
    # @async perception(cam_channel, localization_state_channel, perception_state_channel)
    # @async decision_making(localization_state_channel, perception_state_channel, gt_state_channel,  map, socket)
    #@async
    decision_making(localization_state_channel, nothing, gt_channel,  map_segments, target_map_segment, socket)
end
