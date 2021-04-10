import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mach import MachSolver, Mesh, Vector

options = {
    "silent": False,
    "print-options": False,
    "mesh": {
        "file": "mesh/motor.smb",
        "model-file": "mesh/motor.egads"
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
        "type": "hypregmres",
        "printlevel": 2,
        "maxiter": 125,
        "abstol": 0.0,
        "reltol": 1e-8
    },
    "lin-prec": {
        "type": "hypreams",
        "printlevel": 0
    },
    "nonlin-solver": {
        "type": "relaxed-newton",
        "printlevel": 3,
        "maxiter": 5,
        "reltol": 1e-4,
        "abstol": 5e-1,
        "abort": False
    },
    "components": {
        "farfields": {
            "material": "air",
            "linear": True,
            "attrs": [1, 2, 3]
        },
        "stator": {
            "attr": 4,
            "material": "hiperco50",
            "linear": True
        },
        "rotor": {
            "attr": 5,
            "material": "hiperco50",
            "linear": True
        },
        "airgap": {
            "attr": 6,
            "material": "air",
            "linear": True
        },
        "magnets": {
            "material": "Nd2Fe14B",
            "linear": True,
            "attrs": [7, 11, 15, 19, 23, 27, 31, 35, 39, 43,
                      8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
                      9, 13, 17, 21, 25, 29, 33, 37, 41, 45,
                      10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
        },
        "windings": {
            "material": "copperwire",
            "linear": True,
            "attrs": [47, 48, 49, 50,
                    51, 52, 53, 54,
                    55, 56, 57, 58,
                    59, 60, 61, 62,
                    63, 64, 65, 66,
                    67, 68, 69, 70,
                    71, 72, 73, 74,
                    75, 76, 77, 78,
                    79, 80, 81, 82,
                    83, 84, 85, 86,
                    87, 88, 89, 90,
                    91, 92, 93, 94,
                    95, 96, 97, 98,
                    99, 100, 101, 102,
                    103, 104, 105, 106,
                    107, 108, 109, 110,
                    111, 112, 113, 114,
                    115, 116, 117, 118,
                    119, 120, 121, 122,
                    123, 124, 125, 126,
                    127, 128, 129, 130,
                    131, 132, 133, 134,
                    135, 136, 137, 138, 139, 140,
                    141, 142, 143, 144, 145, 146]
        }
    },
    "problem-opts": {
        "fill-factor": 1.0,
        "current-density": 1.0,
        "current" : {
            "Phase-A": [47, 48, 49, 50,
                        51, 52, 53, 54,
                        67, 68, 69, 70,
                        75, 76, 77, 78,
                        91, 92, 93, 94,
                        95, 96, 97, 98,
                        111, 112, 113, 114,
                        119, 120, 121, 122],
            "Phase-B": [55, 56, 57, 58,
                        71, 72, 73, 74,
                        87, 88, 89, 90,
                        99, 100, 101, 102,
                        115, 116, 117, 118,
                        131, 132, 133, 134,
                        135, 136, 137, 138, 139, 140,
                        141, 142, 143, 144, 145, 146],
            "Phase-C": [59, 60, 61, 62,
                        63, 64, 65, 66,
                        79, 80, 81, 82,
                        83, 84, 85, 86,
                        103, 104, 105, 106,
                        107, 108, 109, 110,
                        123, 124, 125, 126,
                        127, 128, 129, 130]
        },
        "magnets": {
            "south": [7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
            "cw": [8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
            "north": [9, 13, 17, 21, 25, 29, 33, 37, 41, 45],
            "ccw": [10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
        }
    },
    "bcs": {
        "essential": [1, 3]
    }
}

if __name__ == "__main__":
    solver = MachSolver("Magnetostatic", options)
    solver.createOutput("ACLoss");
    solver.createOutput("DCLoss");

    state = solver.getNewField()
    zero = Vector(np.array([0.0, 0.0, 0.0]))
    solver.setFieldValue(state, zero);


    current_density = 11e6 # 11 A/m^2
    fill_factor = 1.0
    inputs = {
        "current-density": current_density,
        "fill-factor": fill_factor,
        "state": state
    }
    solver.solveForState(inputs, state)

    torque_options = {
        "attributes": [4,
                        7, 11, 15, 19, 23, 27, 31, 35, 39, 43,
                        8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
                        9, 13, 17, 21, 25, 29, 33, 37, 41, 45,
                        10, 14, 18, 22, 26, 30, 34, 38, 42, 46],
        "axis": [0.0, 0.0, 1.0],
        "about": [0.0, 0.0, 0.0]
    }
    solver.createOutput("torque", torque_options);

    torque = solver.calcOutput("torque", inputs)
    print("Torque: ", torque)

    # dc_inputs = {
    #     "fill-factor": fill_factor,
    #     "current-density": current_density,
    #     "state": state
    # }
    # dcloss = solver.calcOutput("DCLoss", dc_inputs);
    # print("DC loss: ", dcloss)

    # r_s = 0.00020245 # m, 26 AWG

    # nsamples = 9
    # freqs = np.linspace(100, 2000, nsamples)
    # fem_ac = np.zeros(nsamples)

    # for i in range(nsamples):
    #     # freq = 1e3 # 1000 Hz
    #     freq = float(freqs[i])

    #     ac_inputs = {
    #         "diam": r_s*2,
    #         "frequency": freq,
    #         "fill-factor": fill_factor,
    #         "state": state
    #     }
    #     acloss = solver.calcOutput("ACLoss", ac_inputs);
    #     print("FEM AC loss: ", acloss)
    #     fem_ac[i] = acloss

    # print(fem_ac)
    # print(freqs)

    # fig, ax = plt.subplots()
    # ax.loglog(freqs, fem_ac, label="Hybrid-FEM")
    # ax.set(xlabel='frequency (Hz)', ylabel='AC Loss (W)')
    # ax.grid()
    # fig.savefig("motor_acloss_loglog.png")

    # fig, ax = plt.subplots()
    # ax.plot(freqs, fem_ac, label="Hybrid-FEM")
    # ax.set(xlabel='frequency (Hz)', ylabel='AC Loss (W)')
    # ax.grid()
    # fig.savefig("motor_acloss.png")



