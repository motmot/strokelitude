from tables import IsDescription, UInt64Col, FloatCol, StringCol

class TimeseriesDescription(IsDescription):
    frame = UInt64Col(pos=0) # the most recent data frame
    trigger_timestamp = FloatCol(pos=1) # when the image for this frame happened
    receive_timestamp = FloatCol(pos=2) # when the image trigger happened
    start_display_timestamp = FloatCol(pos=3) # when the image trigger happened
    stop_display_timestamp = FloatCol(pos=4) # when the image trigger happened
    stripe_angle = FloatCol(pos=5) # degrees
    stripe_vel = FloatCol(pos=6) # degrees/second

class StateDescription(IsDescription):
    timestamp = FloatCol(pos=0)
    state = StringCol(255,pos=1)
