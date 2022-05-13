import openmdao.api as om
import numpy as np
# from mach import omEGADS, omMeshMove, MachSolver, omMachState, omMachFunctional
from motor_em_builder import EMMotorBuilder
from mphys import Multipoint
from omESP import omESP
from scenario_motor import ScenarioMotor

from mach import PDESolver, MeshWarper, MachMeshWarper
from motor_current import MotorCurrent

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
        self.options.declare("winding_options", types=dict, desc=" Options for configuring MotorCurrent")
        self.options.declare("check_partials", default=False)

    def setup(self):
        solver_options = self.options["solver_options"]
        warper_options = self.options["warper_options"]
        winding_options = self.options["winding_options"]
        check_partials=self.options["check_partials"]

        self.add_subsystem("geom",
                           omESP(csm_file="model/motor2D.csm",
                                 egads_file="mesh_motor2D.egads"),
                           promotes_inputs=["*"],
                           promotes_outputs=[("x_surf", "x_em"), "model_depth", "num_slots"])

        self.add_subsystem("convert", om.ExecComp("stator_inner_radius = stator_id / 2"),
                            promotes=["*"])

        num_magnets = winding_options["num_magnets"]
        halbach = winding_options["halbach"]
        num_poles = num_magnets
        if halbach:
            num_poles /= 2
        self.add_subsystem("frequency",
                           om.ExecComp(f"frequency = rpm * {num_poles} / 120"),
                           promotes=["*"])

        em_motor_builder = EMMotorBuilder(solver_options=solver_options,
                                          warper_type="MeshWarper",
                                          warper_options=warper_options,
                                          winding_options=winding_options,
                                          check_partials=check_partials)

        em_motor_builder.initialize(self.comm)

        self.mphys_add_scenario("analysis",
                                ScenarioMotor(em_motor_builder=em_motor_builder))

        self.promotes("analysis",
                      inputs=["*"], 
                      outputs=["average_torque",
                               "ac_loss",
                               "dc_loss",
                               "stator_core_loss",
                               "stator_mass",
                               "max_flux_magnitude:stator",
                               "max_flux_magnitude:winding",
                               "stator_volume",
                               "average_flux_magnitude",
                               "winding_max_peak_flux",
                               "rms_current",
                               "efficiency",
                               "power_out",
                               "power_in"
                               ])

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
        "num_turns": 504,
        "num_strands": 1,
        # "num_turns": 12,
        # "num_strands": 182,
        "num_magnets": num_magnets_true,
        "halbach": True
    }

    problem = om.Problem(name="motor", reports="n2")

    problem.model = Motor(solver_options=solver_options,
                          warper_options=warper_options,
                          winding_options=winding_options)

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
    problem.set_val("rms_current_density", strand_current_density)
    problem.set_val("strand_radius", strand_radius)
    problem.set_val("rpm", 6000)

    # em_state0 = np.load("em_state0.npy")
    # em_state1 = np.load("em_state1.npy")
    # em_state2 = np.load("em_state2.npy")
    # em_state3 = np.load("em_state3.npy")
    # em_state4 = np.load("em_state4.npy")
    # em_state5 = np.load("em_state5.npy")
    # em_state6 = np.load("em_state6.npy")
    # em_state7 = np.load("em_state7.npy")
    # problem.set_val("analysis.em_state0", em_state0)
    # problem.set_val("analysis.em_state1", em_state1)
    # problem.set_val("analysis.em_state2", em_state2)
    # problem.set_val("analysis.em_state3", em_state3)
    # problem.set_val("analysis.em_state4", em_state4)
    # problem.set_val("analysis.em_state5", em_state5)
    # problem.set_val("analysis.em_state6", em_state6)
    # problem.set_val("analysis.em_state7", em_state7)

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
    # print(f"rotor core loss: {problem.get_val('rotor_core_loss')}")
    # print(f"magnet core loss: {problem.get_val('magnet_core_loss')}")

    peak_flux = problem.get_val("analysis.peak_flux")
    print(f"max val from peak flux: {np.max(peak_flux)}")
    # np.save("motor_model_peak_flux", peak_flux)

    # np.save("em_state0", problem.get_val("analysis.em_state0"))
    # np.save("em_state1", problem.get_val("analysis.em_state1"))
    # np.save("em_state2", problem.get_val("analysis.em_state2"))
    # np.save("em_state3", problem.get_val("analysis.em_state3"))
    # np.save("em_state4", problem.get_val("analysis.em_state4"))
    # np.save("em_state5", problem.get_val("analysis.em_state5"))
    # np.save("em_state6", problem.get_val("analysis.em_state6"))
    # np.save("em_state7", problem.get_val("analysis.em_state7"))

    # motor_model_peak_flux = np.load("motor_model_peak_flux.npy")

    # solver_options.update(solver_options["multipoint"][0])
    # solver = PDESolver(type="magnetostatic",
    #                    solver_options=solver_options)
    # state = np.zeros(solver.getStateSize())

    # core_loss_opts = {
    #     "attributes": [1]
    # }
    # solver.createOutput("core_loss", core_loss_opts)

    # stack_length = 0.0345
    # model_depth = 0.00075
    # inputs = {
    #     "peak_flux": motor_model_peak_flux,
    #     "frequency": 1000
    # }
    # core_loss = solver.calcOutput("core_loss", inputs) * stack_length / model_depth
    # print(f"Core loss: {core_loss}")

    # mass_opts = {
    #     "attributes": [1]
    # }
    # solver.createOutput("mass", mass_opts)
    # mass = solver.calcOutput("mass", {"state": state}) * stack_length / model_depth
    # print(f"Mass: {mass}")

    # # winding config inputs
    # num_turns = 504
    # num_strands_in_hand = 1
    # # num_turns = 42
    # # num_strands_in_hand = 12
    # strand_radius = 0.00016
    # ac_loss_opts = {
    #     "num_strands": num_strands_in_hand * num_turns
    # }
    # solver.createOutput("ac_loss", ac_loss_opts)

    # inputs = {
    #     "peak_flux": motor_model_peak_flux,
    #     "strand_radius": strand_radius,
    #     "frequency": 1000,
    #     # "num_sih": num_strands_in_hand,
    #     # "num_turns": num_turns,
    #     # "num_strands": num_strands_in_hand * num_turns,
    #     # "slot_area": 7.29100258e-05 * 0.001,
    #     # "length": 0.0345,
    #     "stack_length": 0.0345
    # }

    # ac_losses = []
    # frequencies = np.linspace(100, 1000, 10)
    # for freq in frequencies:
    #     inputs["frequency"] = np.array([freq])
    #     ac_losses.append(solver.calcOutput("ac_loss", inputs))

    # print("AC loss: ", ac_losses)

# AC loss:  [7.304261091221271e-06, 2.9217044364885085e-05, 6.573834982099074e-05, 0.00011686817745954034, 0.00018260652728052933, 0.00026295339928396295, 0.00035790879346983914, 0.00046747270983816137, 0.0005916451483889156, 0.0007304261091221173]

# Inexact:
# Newton iteration 19 : ||r|| = 1.17049, ||r||/||r_0|| = 9.45566e-05
# Loss: 3.03791e-09
# volume: 2.67732e-06
# loss / volume: 0.00113468
# avg torque: [27.85121766]
# ac loss: [0.00113468]
# rms current density: [11000000.]
# stator core loss: [163.30834931]
# stator max flux: [2.59540581]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756621]
# winding max flux: [0.63314111]
# winding peak max flux: [0.6265504]
# max val from peak flux: 2.8762050195638076
#  EGADS Info: 0 Objects, 0 Reference in Use (of 40721) at Close!
# python motor_model.py  2333.87s user 36.67s system 73% cpu 53:35.53 total

# Exact:
# Newton iteration 12 : ||r|| = 1.20931, ||r||/||r_0|| = 9.76929e-05
# Loss: 3.03793e-09
# volume: 2.67732e-06
# loss / volume: 0.00113469
# avg torque: [27.85129506]
# ac loss: [0.00113469]
# rms current density: [11000000.]
# stator core loss: [163.30834931]
# stator max flux: [2.59540186]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756703]
# winding max flux: [0.63314712]
# winding peak max flux: [0.62655629]
# max val from peak flux: 2.876205745752801
#  EGADS Info: 0 Objects, 0 Reference in Use (of 40721) at Close!
# python motor_model.py  2267.04s user 35.76s system 61% cpu 1:02:45.01 total

# Exact 0, 2, 4, 6
# Newton iteration 10 : ||r|| = 1.0762, ||r||/||r_0|| = 8.69384e-05
# avg torque: [27.94331743]
# ac loss: [1.98385664]
# rms current density: [11000000.]
# stator core loss: [163.30834931]
# stator max flux: [2.59540186]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756703]
# winding max flux: [0.63314712]
# winding peak max flux: [0.62939175]
# max val from peak flux: 2.87620645483263
# python motor_model.py  970.78s user 27.83s system 110% cpu 15:00.46 total

# Exact 0, 1, 2, 3, 4, 5, 6, 7
# avg torque: [27.85129761]
# ac loss: [2.15020163]
# rms current density: [11000000.]
# stator core loss: [163.30834931]
# stator max flux: [2.59540186]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756703]
# winding max flux: [0.63314712]
# winding peak max flux: [0.62655629]
# max val from peak flux: 2.8762057457521655
#  EGADS Info: 0 Objects, 0 Reference in Use (of 40721) at Close!
# python motor_model.py  1813.20s user 31.69s system 106% cpu 28:55.23 total

# Exact 0, 4
# avg torque: [27.88797048]
# ac loss: [1.60536031]
# rms current density: [11000000.]
# stator core loss: [163.30834931]
# stator max flux: [2.59540186]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756703]
# winding max flux: [0.63314712]
# winding peak max flux: [0.6305208]
# max val from peak flux: 2.826548013437629
#  EGADS Info: 0 Objects, 0 Reference in Use (of 40721) at Close!
# python motor_model.py  560.41s user 24.30s system 120% cpu 8:04.42 total

# Exact 0, 3, 6, 9, 12, 15
# avg torque: [27.84589209]
# ac loss: [2.04381074]
# rms current density: [11000000.]
# stator core loss: [163.30834931]
# stator max flux: [2.59540186]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756703]
# winding max flux: [0.63314712]
# winding peak max flux: [0.62912924]
# max val from peak flux: 2.8524815666920125
#  EGADS Info: 0 Objects, 0 Reference in Use (of 40721) at Close!
# python motor_model.py  1476.67s user 30.06s system 107% cpu 23:15.70 total

# rotations = [0, 4, 8, 12]
# avg torque: [27.88797048]
# ac loss: [1.60536031]
# rms current density: [11000000.]
# stator core loss: [163.30834931]
# stator max flux: [2.59540186]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756703]
# winding max flux: [0.63314712]
# winding peak max flux: [0.6305208]
# max val from peak flux: 2.826548013437629
#  EGADS Info: 0 Objects, 0 Reference in Use (of 40721) at Close!
# python motor_model.py  1158.67s user 29.76s system 35% cpu 55:54.65 total

# rotations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# avg torque: [27.85129761]
# ac loss: [2.15020163]
# rms current density: [11000000.]
# stator core loss: [163.30834931]
# stator max flux: [2.59540186]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756703]
# winding max flux: [0.63314712]
# winding peak max flux: [0.62655629]
# max val from peak flux: 2.8762057457521655
#  EGADS Info: 0 Objects, 0 Reference in Use (of 40721) at Close!
# python motor_model.py  3432.43s user 36.57s system 80% cpu 1:12:04.16 total


# avg torque: [27.94331743]
# ac loss: [1.98385664]
# dc loss: [79.21991341]
# rms current density: [11000000.]
# rms current: [37.15624463]
# stator core loss: [163.30834931]
# stator max flux: [2.59540186]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756703]
# winding max flux: [0.63314712]
# winding peak max flux: [0.62939175]
# max val from peak flux: 2.87620645483263

# avg torque: [27.94331743]
# ac loss: [1.98385664]
# dc loss: [139.69827703]
# rms current density: [11000000.]
# rms current: [37.15624463]
# stator core loss: [163.30834931]
# stator max flux: [2.59540186]
# stator volume: [0.00011714]
# stator mass: [0.9500306]
# airgap avg flux: [0.67756703]
# winding max flux: [0.63314712]
# winding peak max flux: [0.62939175]
# max val from peak flux: 2.87620645483263
