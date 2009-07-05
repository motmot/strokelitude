from tables import IsDescription, UInt16Col, UInt64Col, FloatCol, Float32Col
import motmot.fview_ext_trig.data_format as mdata_format

class StrokelitudeDataDescription(IsDescription):
    frame = UInt64Col(pos=0)
    trigger_timestamp = FloatCol(pos=1) # when the image trigger happened
    processing_timestamp = FloatCol(pos=2) # when the analysis was done
    left = Float32Col(pos=3) # angle, degrees
    right = Float32Col(pos=4) # angle, degrees

AnalogInputWordstreamDescription = mdata_format.AnalogInputWordstreamDescription
TimeDataDescription = mdata_format.TimeDataDescription
