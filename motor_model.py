import openmdao.api as om
import numpy as np
from mach import omEGADS, omMeshMove, MachSolver, omMachState, omMachFunctional
from motor_current import MotorCurrent

class Motor(om.Group): 
    def initialize(self): 
        self.options.declare("meshmove_options", types=dict)
        self.options.declare("em_options", types=dict)
        self.options.declare("torque_options", types=dict)
        self.options.declare("num_turns", types=int, desc=" The number of turns of wire")
        self.options.declare("num_slots", types=int, desc=" The number of teeth in the stator")

    def setup(self):        
        meshmove_options = self.options["meshmove_options"]
        em_options = self.options["em_options"]
        torque_options = self.options["torque_options"]
        num_turns = self.options["num_turns"]
        num_slots = self.options["num_slots"]

        self.add_subsystem("surf_mesh_move",
                            omEGADS(csm_file="model/motor2D",
                                    model_file="mesh/motor2D.egads",
                                    mesh_file="mesh/motor2D.smb",
                                    tess_file="mesh/motor2D.eto"),
                            promotes_inputs=["*"],
                            promotes_outputs=["surf_mesh_disp"])

        self.meshMoveSolver = MachSolver("MeshMovement", meshmove_options, self.comm)
        self.add_subsystem("vol_mesh_move", omMeshMove(solver=self.meshMoveSolver),
                            promotes_inputs=["surf_mesh_disp"],
                            promotes_outputs=["vol_mesh_coords"])
        # self.connect("surf_mesh_move.surf_mesh_disp", "vol_mesh_move.surf_mesh_disp")

        self.add_subsystem("convert", om.ExecComp("stator_inner_radius = stator_id / 2"),
                            promotes=["*"])

        self.add_subsystem("current",
                            MotorCurrent(num_slots=num_slots,
                                         num_turns=num_turns),
                            promotes_inputs=["*"],
                            promotes_outputs=["current_density:phaseA",
                                              "current_density:phaseB",
                                              "current_density:phaseC",
                                              "rms_current"])


        self.emSolver = MachSolver("Magnetostatic", em_options, self.comm)
        self.add_subsystem("em_solver",
                            omMachState(solver=self.emSolver, 
                                        initial_condition=np.array([0.0, 0.0, 0.0]),
                                        depends=["current_density:phaseA",
                                                 "current_density:phaseB",
                                                 "current_density:phaseC",
                                                 "mesh_coords"]),
                            promotes_inputs=["current_density:phaseA",
                                             "current_density:phaseB",
                                             "current_density:phaseC",
                                             ("mesh_coords", "vol_mesh_coords")],
                            promotes_outputs=["state"])
        # self.connect("vol_mesh_move.vol_mesh_coords", "em_solver.mesh-coords")

        self.add_subsystem("torque",
                            omMachFunctional(solver=self.emSolver,
                                             func="torque",
                                             depends=["mesh_coords", "state"],
                                             options=torque_options),
                            promotes_inputs=[("mesh_coords", "vol_mesh_coords"), "state"],
                            promotes_outputs=["torque"])

        # self.add_subsystem("ac_loss",
        #                     omMachFunctional(solver=self.emSolver,
        #                                      func="ac_loss",
        #                                      depends=["strand_radius", "frequency", "mesh_coords", "state"],
        #                                      options=torque_options),
        #                     promotes_inputs=["strand_radius", "frequency", ("mesh_coords", "vol_mesh_coords"), "state"],
        #                     promotes_outputs=["ac_loss"])

        # self.add_subsystem("dc_loss",
        #                     omMachFunctional(solver=self.emSolver,
        #                                      func="dc_loss",
        #                                      depends=["strand_radius", "mesh_coords", "state"],
        #                                      options=torque_options),
        #                     promotes_inputs=[("mesh_coords", "vol_mesh_coords"), "state"],
        #                     promotes_outputs=["dc_loss"])

        # self.connect("vol_mesh_move.vol_mesh_coords", "torque.mesh-coords")
        # self.connect("em_solver.state", "torque.state")

meshmove_options = {
    "mesh": {
        "file": "mesh/motor2D.smb",
        "model-file": "mesh/motor2D.egads",
        "refine": 0
    },
    "print-options": True,
    "space-dis": {
        "degree": 1,
        "basis-type": "H1"
    },
    "time-dis": {
        "steady": True,
        "steady-abstol": 1e-8,
        "steady-reltol": 1e-8,
        "ode-solver": "PTC",
        "t-final": 100,
        "dt": 1e12,
        "cfl": 1.0,
        "res-exp": 2.0
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
    "saveresults": False,
    "problem-opts": {
        "keep-bndrys": "all",
        "uniform-stiff": {
            "lambda": 1,
            "mu": 1
        }
    }
}

num_magnets_true = 40
num_magnets = 40
mag_pitch = num_magnets // num_magnets_true
num_slots = 24

rotation = 0
magnets = [5+2*num_slots + (rotation+i)%num_magnets for i in range(0, num_magnets)]
# north = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(0, num_magnets_true, 2)] for num in subl]
# south = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(1, num_magnets_true, 2)] for num in subl]

south = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(0, num_magnets_true, 4)] for num in subl]
cw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(1, num_magnets_true, 4)] for num in subl]
north = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(2, num_magnets_true, 4)] for num in subl]
ccw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(3, num_magnets_true, 4)] for num in subl]

em_options = {
    "silent": False,
    "print-options": True,
    "mesh": {
        "file": "mesh/motor2D.smb",
        "model-file": "mesh/motor2D.egads",
        "refine": 0
    },
    "space-dis": {
        "basis-type": "nedelec",
        "degree": 1
    },
    "time-dis": {
        "steady": True,
        "steady-abstol": 0.0,
        "steady-reltol": 0.0,
        "ode-solver": "PTC",
        "t-final": 100,
        "dt": 1,
        "max-iter": 8
    },
    "lin-solver": {
        "type": "minres",
        "printlevel": 1,
        "maxiter": 200,
        "abstol": 0.0,
        "reltol": 1e-10
    },
    "lin-prec": {
        "type": "hypreams",
        "printlevel": 0
    },
    "nonlin-solver": {
        "type": "newton",
        "printlevel": 3,
        "maxiter": 25,
        "reltol": 1e-4,
        "abstol": 0.0,
        "abort": False
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
    "problem-opts": {
        "current" : {
            "z": [43, 46, 47, 50, 8, 9, 12, 13, 19, 22, 23, 26, 32, 33, 36, 37],
            "-z": [44, 45, 48, 49, 7, 10, 11, 14, 20, 21, 24, 25, 31, 34, 35, 38]
        },
        "magnets": {
            "north": north,
            "cw": cw,
            "south": south,
            "ccw": ccw
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
    "magnets": {
        "Nd2Fe14B": {
            "north": north,
            "cw": cw,
            "south": south,
            "ccw": ccw
        }
    },
    "bcs": {
        "essential": "all"
    },
}

torque_options = {
    "attributes": [2] + magnets,
    "axis": [0.0, 0.0, 1.0],
    "about": [0.0, 0.0, 0.0]
}

if __name__ == "__main__":
    problem = om.Problem()
    problem.model = Motor(meshmove_options=meshmove_options,
                          em_options=em_options,
                          torque_options=torque_options,
                          num_turns=12,
                          num_slots=24)

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
    problem.model.set_input_defaults("stack_length", 0.001)
    problem.model.set_input_defaults("rotor_rotation", -4.5)
    problem.model.set_input_defaults("shoe_spacing", 0.0035)
    problem.model.set_input_defaults("num_strands", 42)
    problem.model.set_input_defaults("strand_radius", 0.00016)
    problem.model.set_input_defaults("rms_current_density", 11e6)
    problem.model.set_input_defaults("frequency", 1000)
    problem.setup()

    problem.run_model()

    print("current_density:phaseA", problem.get_val("current_density:phaseA"))
    print("current_density:phaseB", problem.get_val("current_density:phaseB"))
    print("current_density:phaseC", problem.get_val("current_density:phaseC"))
    print("rms_current", problem.get_val("rms_current"))
    print("torque: ", problem.get_val("torque")*34.5)
    print("ac_loss: ", problem.get_val("ac_loss"))
    print("slot_area: ", problem.get_val("slot_area"))
    print("copper_area: ", problem.get_val("copper_area"))
    # print("dc_loss: ", problem.get_val("dc_loss"))
