import parameters

# close haul: going at an angle against the wind
def close_haul():
    parameters.sail_camber = 5000
    parameters.biru_rig_rotation = 25
    parameters.biru_sail_angle = -45
    parameters.kuning_rig_rotation = 15
    parameters.kuning_sail_angle = -45

# beam reach: going perpendicular to the wind    
def beam_reach():
    parameters.sail_camber = 3000
    parameters.biru_rig_rotation = 50
    parameters.biru_sail_angle = -30
    parameters.kuning_rig_rotation = 45
    parameters.kuning_sail_angle = -30

# goose winged: going with the wind from behind,
# with the sails spread in opposite directions
def goose_wing():
    parameters.sail_camber = 10000
    parameters.biru_rig_rotation = 90
    parameters.biru_sail_angle = 45
    parameters.kuning_rig_rotation = 90
    parameters.kuning_sail_angle = -45

# use this setting to play with the sails and masts
def play():
    parameters.sail_camber = 10000
    parameters.biru_rig_rotation = 0
    parameters.biru_sail_angle = 0
    parameters.kuning_rig_rotation = 0
    parameters.kuning_sail_angle = 0

    
