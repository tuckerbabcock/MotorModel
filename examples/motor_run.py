import openmdao.api as om
import numpy as np

from MotorModel import Motor

def _buildSolverOptions(num_magnets_true, num_magnets, num_slots, rotations):
    mag_pitch = num_magnets // num_magnets_true
    multipoint_opts = []
    for rotation in rotations:

        theta_m = rotation / num_magnets * (2*np.pi)
        # divided by 4 since magnetis are in a hallbach array, takes 4 magnets to get 2 poles
        theta_e = (num_magnets_true/2) * theta_m / 2;

        magnets = [5+2*num_slots + (rotation+i)%num_magnets for i in range(0, num_magnets)]
        south = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(0, num_magnets_true, 4)] for num in subl]
        cw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(1, num_magnets_true, 4)] for num in subl]
        north = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(2, num_magnets_true, 4)] for num in subl]
        ccw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(3, num_magnets_true, 4)] for num in subl]

        multipoint_opts.append({
            "magnets": {
                "Nd2Fe14B" : {
                    "north": north,
                    "cw": cw,
                    "south": south,
                    "ccw": ccw
                }
            },
            "theta_e": theta_e
        })

    return multipoint_opts

if __name__ == "__main__":

    warper_options = {
        "mesh": {
            "file": "mesh_motor2D.smb",
            "model-file": "mesh_motor2D.egads",
        },
        "space-dis": {
            "degree": 1,
            "basis-type": "H1"
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 2,
            "reltol": 1e-6,
            "abstol": 1e-6
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

    num_magnets_true = 40
    num_magnets = 160
    num_slots = 24
    # rotations = [0, 1, 2, 3, 4, 5, 6, 7]
    rotations = [0, 2, 4, 6]
    # rotations = [0]
    multipoint_opts = _buildSolverOptions(num_magnets_true,
                                          num_magnets,
                                          num_slots,
                                          rotations=rotations)

    solver_options = {
        "mesh": {
            "file": "mesh_motor2D.smb",
            "model-file": "mesh_motor2D.egads",
        },
        "space-dis": {
            "basis-type": "nedelec",
            "degree": 1
        },
        "time-dis": {
            "steady": True,
        },
        "nonlin-solver": {
            # "type": "inexactnewton",
            "type": "newton",
            "printlevel": 1,
            "maxiter": 15,
            "reltol": 1e-4,
            "abstol": 1.1,
            # "abstol": 1e10,
            "abort": False
        },
        "lin-solver": {
            "type": "minres",
            # "type": "gmres",
            "printlevel": 1,
            "maxiter": 200,
            "abstol": 1e-10,
            "reltol": 1e-10
        },
        "adj-solver": {
            "type": "minres",
            # "type": "gmres",
            "printlevel": 1,
            "maxiter": 200,
            "abstol": 1e-10,
            "reltol": 1e-10
        },
        "lin-prec": {
            "printlevel": -1
        },
        "components": {
            "stator": {
                "attr": 1,
                "material": "hiperco50",
                "linear": False
            },
            "rotor": {
                "attr": 2,
                "material": "hiperco50",
                "linear": False
            },
            "air": {
                "attrs": [3, 4],
                "material": "air",
                "linear": True
            },
            "windings": {
                "material": "copperwire",
                "linear": True,
                "attrs": list(range(5, 5+2*num_slots))
            },
            "magnets": {
                "material": "Nd2Fe14B",
                "linear": True,
                "attrs": list(range(5+2*num_slots, 5+2*num_slots+num_magnets))
            }
        },
        "current": {
            "phaseA": {
                "z": [5, 15, 18, 28, 29, 39, 42, 52],
                "-z": [6, 16, 17, 27, 30, 40, 41, 51]
            },
            "phaseB": {
                "z": [7, 10, 20, 21, 31, 34, 44, 45],
                "-z": [8, 9, 19, 22, 32, 33, 43, 46]
            },
            "phaseC": {
                "z": [12, 13, 23, 26, 36, 37, 47, 50],
                "-z": [11, 14, 24, 25, 35, 38, 48, 49]
            }
        },
        "multipoint": multipoint_opts,
        "bcs": {
            "essential": "all"
        }
    }
    # winding config options
    winding_options = {
        # "num_turns": 504,
        # "num_strands": 1,
        "num_magnets": num_magnets_true,
        "halbach": True
    }

    esp_files = {
        "csm_file": "../model/motor2D.csm",
        "egads_file": "mesh_motor2D.egads"
    }

    problem = om.Problem(name="motor", reports="n2")

    problem.model = Motor(solver_options=solver_options,
                          warper_options=warper_options,
                          winding_options=winding_options,
                          esp_files=esp_files)

    problem.model.set_input_defaults("slot_depth", 0.01110)
    problem.model.set_input_defaults("tooth_tip_thickness", 0.001)
    problem.model.set_input_defaults("tooth_tip_angle", 10.0)
    problem.model.set_input_defaults("slot_radius", 0.001)
    problem.model.set_input_defaults("tooth_width", 0.0043)
    problem.model.set_input_defaults("shoe_spacing", 0.0035)
    problem.model.set_input_defaults("stator_id", 0.1245)
    problem.model.set_input_defaults("stack_length", 0.0345)

    # setup the optimization
    problem.driver = om.ScipyOptimizeDriver()
    problem.driver.options["optimizer"] = "SLSQP"

    problem.model.add_design_var("rms_current_density", lower=1e6, upper=15e6, ref0=1e6, ref=11e6)
    problem.model.add_objective("average_torque", ref=-1)

    # problem.setup()
    problem.setup(mode="rev")

    strand_current_density = 11e6
    strand_radius = 0.00016
    problem.set_val("strands_in_hand", 1)
    problem.set_val("num_turns", 504)
    problem.set_val("rms_current_density", strand_current_density)
    problem.set_val("strand_radius", strand_radius)
    problem.set_val("rpm", 6000)

    # # run the optimization
    # problem.run_driver() 

    problem.run_model()

    print(f"Power out: {problem.get_val('power_out')}")
    print(f"efficiency: {problem.get_val('efficiency')}")
    print(f"avg torque: {problem.get_val('average_torque')}")
    print(f"ac loss: {problem.get_val('ac_loss')}")
    print(f"dc loss: {problem.get_val('dc_loss')}")
    print(f"rms current density: {problem.get_val('rms_current_density')}")
    print(f"rms current: {problem.get_val('rms_current')}")
    print(f"stator core loss: {problem.get_val('stator_core_loss')}")
    print(f"stator max flux: {problem.get_val('max_flux_magnitude:stator')}")
    print(f"stator volume: {problem.get_val('stator_volume')}")
    print(f"stator mass: {problem.get_val('stator_mass')}")
    print(f"airgap avg flux: {problem.get_val('average_flux_magnitude')}")
    print(f"winding max flux: {problem.get_val('max_flux_magnitude:winding')}")
    print(f"winding peak max flux: {problem.get_val('winding_max_peak_flux')}")
