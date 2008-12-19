from tables import IsDescription, UInt64Col, FloatCol

class StrokelitudeDataDescription(IsDescription):
    frame = UInt64Col(pos=0)
    trigger_timestamp = FloatCol(pos=1) # when the image trigger happened
    processing_timestamp = FloatCol(pos=2) # when the analysis was done
    left = FloatCol(pos=3) # angle, degrees
    right = FloatCol(pos=4) # angle, degrees
