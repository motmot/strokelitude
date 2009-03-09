normal_gain = 0

# initialize with defined background conditions
stripe_control.closed_loop( gain=normal_gain, offset=0.0, duration_sec=1 )

# initialize with defined background conditions
stripe_control.closed_loop( gain=normal_gain,
                            offset=SinusoidalBias( freq_hz  = 0.1,
                                                   amplitude= 200, # deg/sec
                                                   ),
                            save_sequence=True,
                            duration_sec=30 )

stripe_control.replay_sequence()

