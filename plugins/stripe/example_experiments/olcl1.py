normal_gain = -40
reverse_gain = -normal_gain

n_repeats = 10
all_gains = [normal_gain,reverse_gain]

total_repititions = n_repeats*len(all_gains)

for i,gain in enumerate(all_gains):
    for repeat in range(n_repeats):
        n_done = i*n_repeats + repeat
        stripe_control.set_progress( n_done, total_repititions )

        stripe_control.show_string(
            'gain %s, repeat %d of %d'%(gain,repeat+1,n_repeats))

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

# reset back to normal closed loop
stripe_control.closed_loop( gain=normal_gain )
