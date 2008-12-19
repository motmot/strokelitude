normal_gain = -900
reverse_gain = -normal_gain

for gain in [normal_gain,reverse_gain]:
    for repeat in range(10):

        # 30 seconds closed loop
        stripe_control.closed_loop( gain=gain, offset=0.0, duration_sec=30 )

        # left to right sweep
        stripe_control.open_loop_sweep( start_pos_deg = 180,
                                        stop_pos_deg = -180,
                                        velocity_dps = -200 )

        # 30 seconds closed loop
        stripe_control.closed_loop( gain=gain, offset=0.0, duration_sec=30 )

        # right to left sweep
        stripe_control.open_loop_sweep( start_pos_deg = -180,
                                        stop_pos_deg = 180,
                                        velocity_dps = 200 )

