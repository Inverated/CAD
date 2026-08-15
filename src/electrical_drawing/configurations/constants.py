"""Layout constants for the electrical drawing system.

Distances are in schemdraw drawing units; page sizes are in inches.
"""

# Spacing between adjacent components and between parallel strings.
COMPONENT_DISTANCE = 2

# Vertical separation enforced between an array's output terminals. Matches the
# MPPT pin gap so arrays wire straight into an MPPT without doglegs.
TERMINATING_DISTANCE = 2

# A4 landscape, in inches.
PAGE_WIDTH_INCHES = 11.69
PAGE_HEIGHT_INCHES = 8.27
PAGE_MARGIN_INCHES = 0.5

# Horizontal gap between the panel array terminals and the MPPT input pins.
MPPT_INPUT_LEAD = 2

# Horizontal length of the lead running from an MPPT output pin to its Tag.
TAG_LEAD = 1.5

# Battery bus drawing geometry.
BUS_STUB = 1.5          # lead length from an incoming Tag to the bus bar
BUS_SEGMENT = 3         # horizontal spacing between bus tap points
BUS_RAIL_GAP = 6        # vertical separation between the +ve and -ve bus bars


# Base drawing scale. The page scaler multiplies BASE_INCHES_PER_UNIT and
# BASE_FONTSIZE by the same factor so text stays proportional to geometry and
# the estimated tag widths remain valid.
BASE_INCHES_PER_UNIT = 0.5
BASE_FONTSIZE = 10
TITLE_FONTSIZE = 13

# Tag width estimation (drawing units), calibrated for BASE_FONTSIZE.
TAG_CHAR_WIDTH = 0.28
TAG_PADDING = 0.8
TAG_MIN_WIDTH = 1.5

# Horizontal spacing between parallel strings, widened for arrays whose
# elements carry wide labels.
PANEL_SPACING = 2.5
BATTERY_SPACING = 5

# Loads carry wide name labels, so their output rails are run further right to
# keep the bus tap clear of the text, and successive loads are spaced wider.
LOAD_RAIL = 6
LOAD_SPACING = 8

# Vertical gap between the title and the top of a drawing.
TITLE_OFFSET = 1.2
