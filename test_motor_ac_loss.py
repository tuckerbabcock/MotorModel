import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import openmdao.api as om
import math

from mach import MachSolver
from motor_current import MotorCurrent

num_magnets_true = 40
num_magnets = 160 #40
mag_pitch = num_magnets // num_magnets_true
num_slots = 24

start = 0
nturns = 1

if __name__ == "__main__":
    rotation = 0
    theta_m = rotation / num_magnets * (2*np.pi)

    # divided by 4 since magnetis are in a hallbach array, takes 4 magnets to get 2 poles
    theta_e = (num_magnets_true/2) * theta_m / 2;
    print("theta_m: ", theta_m, "theta_e: ", theta_e)
    magnets = [5+2*num_slots + (rotation+i)%num_magnets for i in range(0, num_magnets)]
    # north = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(0, num_magnets_true, 2)] for num in subl]
    # south = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(1, num_magnets_true, 2)] for num in subl]

    south = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(0, num_magnets_true, 4)] for num in subl]
    cw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(1, num_magnets_true, 4)] for num in subl]
    north = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(2, num_magnets_true, 4)] for num in subl]
    ccw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(3, num_magnets_true, 4)] for num in subl]

    options = {
        "silent": False,
        "print-options": True,
        "mesh": {
            # "file": "mesh/reference_motor/motor2D.smb",
            # "model-file": "mesh/reference_motor/motor2D.egads",
            # "file": "../meshes/ref_120_mags/motor2D.smb",
            # "model-file": "../meshes/ref_120_mags/motor2D.egads",
            # "file": "../meshes/ref_160_mags/motor2D.smb",
            "file": "../meshes/ref_160_mags/parmesh/motor2D.smb",
            "model-file": "../meshes/ref_160_mags/motor2D.egads",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "ND",
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
            "type": "inexactnewton",
            # "type": "newton",
            "printlevel": 1,
            "maxiter": 25,
            "reltol": 1e-4,
            "abstol": 1.1,
            "abort": False
        },
        "components": {
            "stator": {
                "attr": 1,
                "material": "hiperco50",
                "linear": False
                # "linear": True
            },
            "rotor": {
                "attr": 2,
                "material": "hiperco50",
                "linear": False
                # "linear": True
            },
            "aircore": {
                "material": "air",
                "linear": True,
                "attr": 4
            },
            "airgap": {
                "material": "copperwire",
                "linear": True,
                "attr": 3
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
        "ess-bdr": "all",
        "bcs": {
            "essential": "all"
        }
    }
    solver = MachSolver("Magnetostatic", options)

    # state = solver.getNewField()
    # zero = np.array([0.0, 0.0, 0.0])
    # solver.setFieldValue(state, zero);

    # state = np.load("state.npy")
    state = np.zeros(solver.getStateSize())

    # winding config inputs
    # num_turns = 504
    # num_strands_in_hand = 1
    num_turns = 42
    num_strands_in_hand = 12
    strand_radius = 0.00016
    strand_current_density = 11e6

    # slot geometry inputs
    stator_inner_radius = 0.067 #0.06225
    tooth_tip_thickness = 0.00100
    tooth_tip_angle = 10.0000001
    slot_depth = 0.01110
    slot_radius = 0.00100
    tooth_width = 0.005 #0.00430
    shoe_spacing = 0.0035

    problem = om.Problem()
    problem.model.add_subsystem("current",
                                MotorCurrent(num_slots=num_slots,
                                             num_turns=num_turns),
                                promotes_inputs=["*"],
                                promotes_outputs=["current_density:phaseA",
                                                  "current_density:phaseB",
                                                  "current_density:phaseC",
                                                  "rms_current"])
    problem.model.set_input_defaults("rms_current_density", strand_current_density)
    problem.model.set_input_defaults("stator_inner_radius", stator_inner_radius)
    problem.model.set_input_defaults("slot_depth", slot_depth)
    problem.model.set_input_defaults("tooth_tip_thickness", tooth_tip_thickness)
    problem.model.set_input_defaults("tooth_tip_angle", tooth_tip_angle)
    problem.model.set_input_defaults("slot_radius", slot_radius)
    problem.model.set_input_defaults("tooth_width", tooth_width)
    problem.model.set_input_defaults("shoe_spacing", shoe_spacing)
    problem.model.set_input_defaults("strand_radius", strand_radius)
    problem.model.set_input_defaults("num_strands", num_strands_in_hand)
    problem.model.set_input_defaults("theta_e", theta_e)
    problem.setup()
    problem.run_model()

    fill_factor = problem.get_val("fill_factor")
    print("fill factor:", fill_factor)
    # current_density = 7704937.84606975
    # current_density:phaseA =  0.0
    # current_density:phaseB =  7196398.8502466
    # current_density:phaseC = -7196398.85024659
    current_density_phaseA = problem.get_val("current_density:phaseA")
    current_density_phaseB = problem.get_val("current_density:phaseB")
    current_density_phaseC = problem.get_val("current_density:phaseC")
    print("current_density:phaseA", current_density_phaseA)
    print("current_density:phaseB", current_density_phaseB)
    print("current_density:phaseC", current_density_phaseC)

    inputs = {
        "current_density:phaseA": current_density_phaseA,
        "current_density:phaseB": current_density_phaseB,
        "current_density:phaseC": current_density_phaseC,
        "state": state,
    }
    solver.solveForState(inputs, state)
    # np.save("state", state)

    # solver.printField("B_mags", "B")

    torque_options = {
        "attributes": [2] + magnets,
        "axis": [0.0, 0.0, 1.0],
        "about": [0.0, 0.0, 0.0]
    }
    solver.createOutput("torque", torque_options)

    torque = solver.calcOutput("torque", inputs) * 34.5
    print("Torque: ", torque)

    solver.createOutput("flux_magnitude")
    dg_flux_mag_size = solver.getFieldSize("flux_magnitude")
    dg_flux_tv = np.zeros(dg_flux_mag_size)

    solver.calcOutput("flux_magnitude", inputs, dg_flux_tv)
    print("dg_flux_tv", dg_flux_tv)
    print("mag dg_flux", max(dg_flux_tv))
    print("flux size:", dg_flux_mag_size)

    np.save("flux_mag"+str(num_magnets)+"_"+str(rotation), dg_flux_tv)

    # max_flux_options = {
    #     "rho": 500.0,
    #     "attributes": list(range(5, 5+2*num_slots))
    # }
    # solver.createOutput("max_flux", max_flux_options)
    # max_winding_flux = solver.calcOutput("max_flux", inputs);
    # print("Winding max flux: ", max_winding_flux)

    # # max_winding_flux = 0.9;

    # winding_attrs = list(range(5, 5+2*num_slots))
    # avg_flux_options = {
    #     "attributes": [winding_attrs[0]]
    # }
    # solver.createOutput("flux_squared_avg", avg_flux_options)
    # avg_flux = []
    # for attr in winding_attrs:
    #     avg_flux_options["attributes"] = [attr]
    #     solver.setOutputOptions("flux_squared_avg", avg_flux_options)
    #     avg_flux.append(solver.calcOutput("flux_squared_avg", inputs))

    # print(avg_flux)
    # print(max(avg_flux))

    # max_avg_flux = max(avg_flux)

    # ac_loss = 0.0345 * num_strands_in_hand * num_turns * (strand_radius**4) * (900**2) * max_avg_flux * 58.14e6 * np.pi / 32.0
    # print("ac loss: ", ac_loss)

    # solver.createOutput("dc_loss")
    # inputs["fill-factor"] = 1.0
    # dc_loss = solver.calcOutput("dc_loss", inputs) * 34.5
    # print("DC loss: ", dc_loss)

    # solver.createOutput("ac_loss");
    
    # # radii = np.array([0.00016, 0.00032, 0.00048, 0.00064])
    # # frequencies = np.linspace(1000, 6000, 11)
    # # radii = np.array([0.00016, 0.00032])
    # # frequencies = np.range(100, 1000, 100)

    # # ac_loss = np.zeros([radii.size, frequencies.size])
    # # for i in range(0, radii.size):
    # #     for j in range(0, frequencies.size):
    # #         inputs["strand_radius"] = np.array([radii[i]])
    # #         inputs["frequency"] = np.array([frequencies[j]])
    # #         inputs["num_strands"] = 42*12
    # #         inputs["slot_area"] = 7.29100258e-05 * 0.001;
    # #         inputs["length"] = 0.0345;
    # #         ac_loss[i,j] = solver.calcOutput("ac_loss", inputs) * 34.5

    # slot_area = problem.get_val("slot_area")
    # print("slot area:", slot_area)
    # ac_loss = []
    # # for frequency in range(0, 1100, 100):
    # frequency = 900
    # # inputs["max_flux"] = max_winding_flux
    # inputs["max_flux"] = math.sqrt(max_avg_flux)
    # inputs["strand_radius"] = strand_radius
    # inputs["frequency"] = frequency
    # inputs["num_sih"] = num_strands_in_hand
    # inputs["num_turns"] = num_turns
    # inputs["slot_area"] = slot_area # 7.29100258e-05 # 0.0001458
    # inputs["length"] = 0.0345;
    # ac_loss.append(solver.calcOutput("ac_loss", inputs) * 1000 * 2./3)

    # print("AC loss:", ac_loss)
    # print("Frequencies: ", frequencies)

    # rpm = 120 * frequencies / num_magnets_true

    # # Contour plot
    # fig, ax = plt.subplots(nrows=1)
    # cf = ax.contourf(rpm, radii, ac_loss)
    # fig.colorbar(cf)
    # ax.set_xlabel("RPM")
    # ax.set_ylabel("Strand Radius (m)")
    # ax.set_title("AC Loss (W)")
    # plt.show()

    # plt.xlabel("Strand Radius (W)")
    # plt.ylabel("Electrical Frequency (Hz)")
    # plt.plot(frequencies, ac_loss[0, :])
    # plt.ylabel("AC Loss (W)")
    # plt.xlabel("Electrical Frequency (Hz)")
    # plt.show()

    # plt.plot(range(0, 1100, 100), ac_loss)
    # plt.show()

# rms_current [37.15624463]
# torque:  [-32.48443897]
# ac_loss:  [0.00058583]
# current_density [7704937.84606975]
# rms_current [37.15624463]
# torque:  [-32.48443897]
# ac_loss:  [0.00058583]
# slot_area:  [7.29100258e-05]
# copper_area:  [4.05340851e-05]