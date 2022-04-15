import openmdao.api as om
import numpy as np
# from mach import omEGADS, omMeshMove, MachSolver, omMachState, omMachFunctional
from motor_em_builder import EMMotorBuilder
from mphys import Multipoint
from omESP import omESP
from scenario_motor import ScenarioMotor

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

class Motor(Multipoint): 
    def initialize(self): 
        self.options.declare("solver_options", types=dict)
        self.options.declare("warper_options", types=dict)
        self.options.declare("outputs", types=dict, default=None)
        self.options.declare("winding_options", types=dict, desc=" Options for configuring MotorCurrent")
        self.options.declare("check_partials", default=False)

    def setup(self):
        solver_options = self.options["solver_options"]
        warper_options = self.options["warper_options"]
        outputs = self.options["outputs"]
        winding_options = self.options["winding_options"]
        check_partials=self.options["check_partials"]

        self.add_subsystem("geom",
                           omESP(csm_file="model/motor2D.csm",
                                 egads_file="mesh_motor2D.egads"),
                           promotes_inputs=["*"],
                           promotes_outputs=[("x_surf", "x_em")])

        self.add_subsystem("convert", om.ExecComp("stator_inner_radius = stator_id / 2"),
                            promotes=["*"])

        depends = ["current_density:phaseA",
                   "current_density:phaseB",
                   "current_density:phaseC"]

        em_motor_builder = EMMotorBuilder(solver_options=solver_options,
                                          depends=depends,
                                          warper_type="MeshWarper",
                                          warper_options=warper_options,
                                          outputs=outputs,
                                          winding_options=winding_options,
                                          check_partials=check_partials)

        em_motor_builder.initialize(self.comm)

        self.mphys_add_scenario('analysis',
                                ScenarioMotor(em_motor_builder=em_motor_builder))

        self.promotes("analysis", inputs=["*"])

        # promote all state inputs
        # self.promotes("analysis", inputs=["*"])
        # promote all analysis output inputs
        # for output in outputs:
        #     if "depends" in outputs[output]:
        #         self.promotes("analysis", inputs=[*outputs[output]["depends"]])

        # self.connect("geom.x_surf", "analysis.x_em")

if __name__ == "__main__":

    warper_options = {
        "mesh": {
            # "file": "mesh/motor2D.smb",
            # "model-file": "mesh/motor2D.egads",
            "file": "mesh_motor2D.smb",
            "model-file": "mesh_motor2D.egads",
        },
        "space-dis": {
            "degree": 1,
            "basis-type": "H1"
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 3,
            "maxiter": 5,
            "reltol": 1e-12,
            "abstol": 1e-12
        },
        "lin-solver": {
            "type": "hyprepcg",
            "printlevel": -1,
            "maxiter": 100,
            "abstol": 1e-10,
            "reltol": 1e-10
        },
        "lin-prec": {
            "type": "hypreboomeramg",
            "printlevel": -1
        },
    }

    num_magnets_true = 40
    num_magnets = 160
    num_slots = 24

    multipoint_opts = _buildSolverOptions(num_magnets_true,
                                          num_magnets,
                                          num_slots,
                                        #   rotations=[0, 2, 4, 6])
                                          rotations=[0, 2])

    solver_options = {
        "mesh": {
            # "file": "mesh/motor2D.smb",
            # "model-file": "mesh/motor2D.egads",
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
            "maxiter": 100,
            "reltol": 1e-4,
            "abstol": 1.1,
            "abort": False
        },
        "lin-solver": {
            # "type": "minres",
            "type": "gmres",
            "printlevel": 1,
            "maxiter": 200,
            "abstol": 1e-10,
            "reltol": 1e-10
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
            "printlevel": -1
        },
        "components": {
            "stator": {
                "attr": 1,
                "material": "hiperco50",
                "linear": True
            },
            "rotor": {
                "attr": 2,
                "material": "hiperco50",
                "linear": True
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

    outputs = {
        "torque": {
            "options": {
                "attributes": [2] + list(range(5+2*num_slots, 5+2*num_slots+num_magnets)),
                "axis": [0.0, 0.0, 1.0],
                "about": [0.0, 0.0, 0.0]
            }
        },
        "ac_loss": {
            "depends": [
                "stack_length",
                "slot_area",
                "frequency",
                "peak_flux",
                "strand_radius"
            ]
        }
    }

    winding_options = {
        "num_slots": num_slots,
        "num_turns": 12,
        "num_strands": 42
    }

    problem = om.Problem()
    problem.model = Motor(solver_options=solver_options,
                          warper_options=warper_options,
                          outputs=outputs,
                          winding_options=winding_options)

    problem.model.set_input_defaults("stator_od", 0.15645)
    problem.model.set_input_defaults("stator_id", 0.12450)
    problem.model.set_input_defaults("rotor_od", 0.11370)
    problem.model.set_input_defaults("rotor_id", 0.11125)
    problem.model.set_input_defaults("slot_depth", 0.01110)
    problem.model.set_input_defaults("tooth_width", 0.00430)
    problem.model.set_input_defaults("magnet_thickness", 0.00440)
    problem.model.set_input_defaults("heatsink_od", 0.16000)
    problem.model.set_input_defaults("tooth_tip_thickness", 0.00100)
    problem.model.set_input_defaults("tooth_tip_angle", 10.0000001)
    problem.model.set_input_defaults("slot_radius", 0.00100)
    problem.model.set_input_defaults("stack_length", 0.0345)
    # problem.model.set_input_defaults("rotor_rotation", -4.5)
    problem.model.set_input_defaults("shoe_spacing", 0.0035)
    # problem.model.set_input_defaults("num_strands", 42)
    # problem.model.set_input_defaults("strand_radius", 0.00016)
    # problem.model.set_input_defaults("rms_current_density", 11e6)
    # problem.model.set_input_defaults("frequency", 1000)
    problem.setup()

    problem.run_model()

    # print("current_density:phaseA", problem.get_val("current_density:phaseA"))
    # print("current_density:phaseB", problem.get_val("current_density:phaseB"))
    # print("current_density:phaseC", problem.get_val("current_density:phaseC"))
    # print("rms_current", problem.get_val("rms_current"))
    # print("torque: ", problem.get_val("torque"))
    # print("ac_loss: ", problem.get_val("ac_loss"))
    # print("slot_area: ", problem.get_val("slot_area"))
    # print("copper_area: ", problem.get_val("copper_area"))
    # # print("dc_loss: ", problem.get_val("dc_loss"))
