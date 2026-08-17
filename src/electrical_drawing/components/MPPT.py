from schemdraw.elements import Element
from schemdraw.segments import Segment, SegmentText

class MPPT(Element):
    """A 4-terminal MPPT charge controller.

    Anchors: ``PV+``, ``PV-`` on the left (panel side) and ``BATT+``, ``BATT-``
    on the right (battery side), plus ``center``.

    Parameters
    ----------
    pin_gap:
        Vertical distance between the two pins on each side. Match this to the
        terminal separation of the array feeding it so the wires run straight.
    name:
        Text drawn inside the body, e.g. ``"MPPT1"``.
    """

    def __init__(self, pin_gap=2, name='MPPT', **kwargs):
        super().__init__(**kwargs)
        w = 3
        h = pin_gap * 2
        pin = 0.5
        label_offset = 0.7
        
        self.segments.append(
            Segment([
                (0, 0), (w, 0), (w, h), (0, h), (0, 0)
            ])
        )

        # Input pins (left)
        self.segments.append(Segment([(-pin, h*0.75), (0, h*0.75)]))
        self.segments.append(SegmentText((-pin+label_offset, h*0.75), 'PV+', align=['left', 'center']))
        
        self.segments.append(Segment([(-pin, h*0.25), (0, h*0.25)]))
        self.segments.append(SegmentText((-pin+label_offset, h*0.25), 'PV-', align=['left', 'center']))

        # Output pins (right)
        self.segments.append(Segment([(w, h*0.75), (w+pin, h*0.75)]))
        self.segments.append(SegmentText((w+pin-label_offset, h*0.75), 'BATT+', align=['right', 'center']))
        
        self.segments.append(Segment([(w, h*0.25), (w+pin, h*0.25)]))
        self.segments.append(SegmentText((w+pin-label_offset, h*0.25), 'BATT-', align=['right', 'center']))

        self.segments.append(SegmentText((w/2, h/2), name, align=['center', 'center'], fontsize=12))
        
        self.anchors['PV+']  = (-pin, h*0.75)
        self.anchors['PV-']  = (-pin, h*0.25)
        self.anchors['BATT+'] = (w+pin, h*0.75)
        self.anchors['BATT-'] = (w+pin, h*0.25)
        self.anchors['center'] = (w/2, h/2)