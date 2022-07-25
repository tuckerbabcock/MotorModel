import numpy as np

def _buildMultipointOptions(magnet_attrs, magnet_divisions, rotations):

    num_magnet_attrs = len(magnet_attrs)

    magnet_idxs = {
        "south": [i for i in range(0, num_magnet_attrs, 4)],
        "cw": [i for i in range(1, num_magnet_attrs, 4)],
        "north": [i for i in range(2, num_magnet_attrs, 4)],
        "ccw": [i for i in range(3, num_magnet_attrs, 4)]
    }

    multipoint_opts = []
    for rotation in rotations:
        rotated_attrs = [magnet_attrs[(i+rotation) % num_magnet_attrs] for i, _ in enumerate(magnet_attrs)]

        theta_m = rotation / num_magnet_attrs * (2*np.pi)
        # divided by 4 since magnetis are in a hallbach array, takes 4 magnets to get 2 poles
        theta_e = ((num_magnet_attrs // magnet_divisions)/2) * theta_m / 2;

        multipoint_opts.append({
            "magnets": {
                "Nd2Fe14B" : {
                    "north": [rotated_attrs[i] for i in magnet_idxs["north"]],
                    "cw": [rotated_attrs[i] for i in magnet_idxs["cw"]],
                    "south": [rotated_attrs[i] for i in magnet_idxs["south"]],
                    "ccw": [rotated_attrs[i] for i in magnet_idxs["ccw"]]
                }
            },
            "theta_e": theta_e
        })
    return multipoint_opts

    # mag_pitch = num_magnets // num_magnets_true
    # multipoint_opts = []
    # for rotation in rotations:

    #     theta_m = rotation / num_magnets * (2*np.pi)
    #     # divided by 4 since magnetis are in a hallbach array, takes 4 magnets to get 2 poles
    #     theta_e = (num_magnets_true/2) * theta_m / 2;

    #     magnets = [4+2*num_slots + (rotation+i)%num_magnets for i in range(0, num_magnets)]
    #     south = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(0, num_magnets_true, 4)] for num in subl]
    #     cw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(1, num_magnets_true, 4)] for num in subl]
    #     north = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(2, num_magnets_true, 4)] for num in subl]
    #     ccw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(3, num_magnets_true, 4)] for num in subl]

    #     multipoint_opts.append({
    #         "magnets": {
    #             "Nd2Fe14B" : {
    #                 "north": north,
    #                 "cw": cw,
    #                 "south": south,
    #                 "ccw": ccw
    #             }
    #         },
    #         "theta_e": theta_e
    #     })

    # return multipoint_opts

def _buildCurrentOptions(current_attrs):
    current_idxs = {
        "phaseA": {
            "z": [0, 10, 13, 23, 24, 34, 37, 47],
            "-z": [1, 11, 12, 22, 25, 35, 36, 46]
        },
        "phaseB": {
            "z": [2, 5, 15, 16, 26, 29, 39, 40],
            "-z": [3, 4, 14, 17, 27, 28, 38, 41]
        },
        "phaseC": {
            "z": [7, 8, 18, 21, 31, 32, 42, 45],
            "-z": [6, 9, 19, 20, 30, 33, 43, 44]
        }
    }

    current_options = {
        "phaseA": {
            "z": [current_attrs[i] for i in current_idxs["phaseA"]["z"]],
            "-z": [current_attrs[i] for i in current_idxs["phaseA"]["-z"]]
        },
        "phaseB": {
            "z": [current_attrs[i] for i in current_idxs["phaseB"]["z"]],
            "-z": [current_attrs[i] for i in current_idxs["phaseB"]["-z"]]
        },
        "phaseC": {
            "z": [current_attrs[i] for i in current_idxs["phaseC"]["z"]],
            "-z": [current_attrs[i] for i in current_idxs["phaseC"]["-z"]]
        }
    }
    return current_options


def _buildSolverOptions(num_magnets, magnet_divisions, num_slots, two_dimensional):
    warper_options = {
        "mesh": {
            "file": "../mesh_motor2D.smb",
            "model-file": "../mesh_motor2D.egads",
        },
        "space-dis": {
            "degree": 1,
            "basis-type": "H1"
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 2,
            "reltol": 1e-8,
            "abstol": 1e-8
        },
        "lin-solver": {
            "type": "pcg",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-10,
            "reltol": 1e-10
        },
        "adj-solver": {
            "type": "pcg",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-10,
            "reltol": 1e-10
        },
        "lin-prec": {
            "printlevel": -1
        },
        "bcs": {
            "essential": "all"
        }
    }

    if two_dimensional:
        basis = "h1"
        prec = "hypreboomeramg"
        stator_attrs = [93, 90, 89, 86, 85, 82, 81, 78, 77, 74, 73, 70, 69, 66, 65, 62, 61, 58, 57, 54, 53, 50, 49, 46, 45, 42, 41, 38, 37, 34, 33, 30, 6, 2, 1, 3, 8, 26, 25, 22, 21, 18, 17, 14, 13, 10, 97, 94]
        rotor_attrs = [178]
        airgap_attrs = [5]
        current_attrs = [11, 96, 95, 92, 91, 88, 87, 84, 83, 80, 79, 76, 75, 72, 71, 68, 67, 64, 63, 60, 59, 56, 55, 52, 51, 48, 47, 44, 43, 40, 39, 36, 35, 32, 31, 29, 7, 4, 9, 28, 27, 24, 23, 20, 19, 16, 15, 12]
        magnet_attrs =  [177, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176]
    else:
        basis = "nedelec"
        prec = "hypreams"
        stator_attrs = [1]
        rotor_attrs = [2]
        airgap_attrs = [3]
        current_attrs = list(range(4, 4+2*num_slots))
        magnet_attrs =  list(range(4+2*num_slots, 4+2*num_slots+num_magnets*magnet_divisions))


    current_options = _buildCurrentOptions(current_attrs)
    # _num_magnets_true = 40
    # _num_magnets = 160
    # _num_slots = 24
    # rotations = [0, 1, 2, 3, 4, 5, 6, 7]
    # rotations = [0, 2, 4, 6]
    # rotations = [0, 1, 2, 3]
    rotations = [1]
    # multipoint_opts = _buildMultipointOptions(magnet_attrs,
    #                                           num_magnets_true,
    #                                           num_magnets,
    #                                           num_slots,
    #                                           rotations)
    multipoint_opts = _buildMultipointOptions(magnet_attrs, magnet_divisions, rotations)



    em_options = {
        "mesh": {
            "file": "../mesh_motor2D.smb",
            "model-file": "../mesh_motor2D.egads",
        },
        "space-dis": {
            "basis-type": basis,
            "degree": 1
        },
        "nonlin-solver": {
            # "type": "inexactnewton",
            "type": "newton",
            "printlevel": 1,
            "maxiter": 15,
            "reltol": 1e-8,
            "abstol": 1e-6,
            "abort": False
        },
        "lin-solver": {
            # "type": "minres",
            "type": "gmres",
            "kdim": 200,
            "printlevel": 1,
            "maxiter": 200,
            "abstol": 1e-12,
            "reltol": 1e-12
        },
        "adj-solver": {
            # "type": "minres",
            "type": "gmres",
            "printlevel": 1,
            "maxiter": 200,
            "abstol": 1e-10,
            "reltol": 1e-10
        },
        "lin-prec": {
            "type": prec,
            "printlevel": -1
        },
        "components": {
            "stator": {
                "attrs": stator_attrs,
                "material": "hiperco50",
                "linear": False
            },
            "rotor": {
                "attrs": rotor_attrs,
                "material": "hiperco50",
                "linear": False
            },
            "airgap": {
                "attrs": airgap_attrs,
                "material": "air",
                "linear": True
            },
            # "aircore": {
            #     "attrs": [4],
            #     "material": "air",
            #     "linear": True
            # },
            # "shaft": {
            #     "attrs": [4],
            #     "material": "air",
            #     "linear": True
            # },
            # "heatsink": {
            #     "attrs": [5],
            #     "material": "air",
            #     "linear": True
            # },
            "windings": {
                "attrs": current_attrs,
                "material": "copperwire",
                "linear": True
            },
            "magnets": {
                "attrs": magnet_attrs,
                "material": "Nd2Fe14B",
                "linear": True
            }
        },
        "current": current_options,
        # {
            # "phaseA": {
            #     "z": [5, 15, 18, 28, 29, 39, 42, 52],
            #     "-z": [6, 16, 17, 27, 30, 40, 41, 51]
            # },
            # "phaseB": {
            #     "z": [7, 10, 20, 21, 31, 34, 44, 45],
            #     "-z": [8, 9, 19, 22, 32, 33, 43, 46]
            # },
            # "phaseC": {
            #     "z": [12, 13, 23, 26, 36, 37, 47, 50],
            #     "-z": [11, 14, 24, 25, 35, 38, 48, 49]
            # }
            # "phaseA": {
            #     "z": [4, 14, 17, 27, 28, 38, 41, 51],
            #     "-z": [5, 15, 16, 26, 29, 39, 40, 50]
            # },
            # "phaseB": {
            #     "z": [6, 9, 19, 20, 30, 33, 43, 44],
            #     "-z": [7, 8, 18, 21, 31, 32, 42, 45]
            # },
            # "phaseC": {
            #     "z": [11, 12, 22, 25, 35, 36, 46, 49],
            #     "-z": [10, 13, 23, 24, 34, 37, 47, 48]
            # }
            # "phaseA": {
            #     "z": [6, 16, 19, 29, 30, 40, 43, 53],
            #     "-z": [7, 17, 18, 28, 31, 41, 42, 52]
            # },
            # "phaseB": {
            #     "z": [8, 11, 21, 22, 32, 35, 45, 46],
            #     "-z": [9, 10, 20, 23, 33, 34, 44, 47]
            # },
            # "phaseC": {
            #     "z": [13, 14, 24, 27, 37, 38, 48, 51],
            #     "-z": [12, 15, 25, 26, 36, 39, 49, 50]
            # }
        # },
        "multipoint": multipoint_opts,
        "bcs": {
            "essential": "all"
        }
    }

    thermal_options = {
        "mesh": {
            "file": "../mesh_motor2D.smb",
            "model-file": "../mesh_motor2D.egads",
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 2,
            "reltol": 1e-10,
            "abstol": 1e-10,
            "abort": True
        },
        "lin-solver": {
            "type": "pcg",
            # "type": "gmres",
            "printlevel": 1,
            "maxiter": 200,
            "abstol": 1e-12,
            "reltol": 1e-12
        },
        "adj-solver": {
            "type": "pcg",
            # "type": "gmres",
            "printlevel": 1,
            "maxiter": 200,
            "abstol": 1e-12,
            "reltol": 1e-12
        },
        "lin-prec": {
            "printlevel": -1
        },
        "components": {
            "stator": {
                "attrs": [1],
                "material": "hiperco50",
            },
            "rotor": {
                "attrs": [2],
                "material": "hiperco50",
            },
            "airgap": {
                "attrs": [3],
                "material": "air",
                "linear": True
            },
            # "air": {
            #     "attrs": [4],
            #     "material": "air",
            #     "linear": True
            # },
            "windings": {
                "material": "copperwire",
                "linear": True,
                "attrs": list(range(4, 4+2*num_slots))
            },
            "magnets": {
                "material": "Nd2Fe14B",
                "linear": True,
                "attrs": list(range(4+2*num_slots, 4+2*num_slots+num_magnets*magnet_divisions))
            }
        },
        "bcs": {
            "convection": [1047, 1060, 1081,
                           1094, 1115, 1128,
                           1149, 1162, 1183,
                           1196, 1217, 1229,
                           1211, 1195, 1177,
                           1161, 1143, 1127,
                           1109, 1093, 1075,
                           1059, 1041, 1025,
                           1007, 991, 973,
                           957, 939, 923,
                           905, 595, 19,
                           3, 1, 5,
                           25, 850, 878,
                           890, 911, 924,
                           945, 958, 979,
                           992, 1013, 1026],
        },
        # "external-fields": {
        #     "thermal_load"
        # }
    }

    return warper_options, em_options, thermal_options


    ### True 2D model attributes
    # "stator": {
    #             "attrs": [93, 90, 89, 86, 85, 82, 81, 78, 77, 74, 73, 70, 69, 66, 65, 62, 61, 58, 57, 54, 53, 50, 49, 46, 45, 42, 41, 38, 37, 34, 33, 30, 6, 2, 1, 3, 8, 26, 25, 22, 21, 18, 17, 14, 13, 10, 97, 94],
    #             "material": "hiperco50",
    #             "linear": False
    #         },

    # "rotor": {
    #             "attrs": [178],
    #             "material": "hiperco50",
    #             "linear": False
    #         },

    # "windings": {
    #             "attrs": [11, 96, 95, 92, 91, 88, 87, 84, 83, 80, 79, 76, 75, 72, 71, 68, 67, 64, 63, 60, 59, 56, 55, 52, 51, 48, 47, 44, 43, 40, 39, 36, 35, 32, 31, 29, 7, 4, 9, 28, 27, 24, 23, 20, 19, 16, 15, 12],
    #             "material": "copperwire",
    #             "linear": True
    #         },

    # "magnets": {
    #             "attrs": [177, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176],
    #             "material": "Nd2Fe14B",
    #             "linear": True
    #         }