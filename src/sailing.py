import parameters

# close haul: going at an angle against the wind
def close_haul():
    parameters.sail_camber = 5000
    parameters.rig_rotation_biru = 25
    parameters.sail_angle_biru = -45
    parameters.rig_rotation_kuning = 15
    parameters.sail_angle_kuning = -45

# beam reach: going perpendicular to the wind    
def beam_reach():
    parameters.sail_camber = 4000
    parameters.rig_rotation_biru = 40
    parameters.sail_angle_biru = -30
    parameters.rig_rotation_kuning = 30
    parameters.sail_angle_kuning = -30

# broad reach: going with the wind from an aft quarter
def broad_reach():
    parameters.sail_camber = 3000
    parameters.rig_rotation_biru = 50
    parameters.sail_angle_biru = -30
    parameters.rig_rotation_kuning = 45
    parameters.sail_angle_kuning = -30

# goose winged: going with the wind from behind,
# with the sails spread in opposite directions
def goose_wing():
    parameters.sail_camber = 10000
    parameters.rig_rotation_biru = 90
    parameters.sail_angle_biru = 45
    parameters.rig_rotation_kuning = 90
    parameters.sail_angle_kuning = -45

# use this setting to play with the sails and masts
def play():
    parameters.sail_camber = 10000
    parameters.rig_rotation_biru = 0
    parameters.sail_angle_biru = 0
    parameters.rig_rotation_kuning = 0
    parameters.sail_angle_kuning = 0

    
