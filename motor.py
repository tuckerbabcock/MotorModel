import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mach import MachSolver, Mesh, Vector

num_magnets_true = 40
num_magnets = 160
mag_pitch = num_magnets // num_magnets_true
num_slots = 24

start = 10
nturns = 1
torque = []

if __name__ == "__main__":
    for rotation in range(start, start+nturns):
    # for rotation in range(nturns, 2*nturns):
        magnets = [7+4*num_slots + (rotation+i)%num_magnets for i in range(0, num_magnets)]
        # north = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(0, num_magnets_true, 2)] for num in subl]
        # south = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(1, num_magnets_true, 2)] for num in subl]

        south = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(0, num_magnets_true, 4)] for num in subl]
        cw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(1, num_magnets_true, 4)] for num in subl]
        north = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(2, num_magnets_true, 4)] for num in subl]
        ccw = [num for subl in [magnets[i*mag_pitch:(i+1)*mag_pitch][:] for i in range(3, num_magnets_true, 4)] for num in subl]

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
                "type": "minres",
                "printlevel": 2,
                "maxiter": 150,
                "abstol": 0.0,
                "reltol": 1e-10
            },
            "lin-prec": {
                "type": "hypreams",
                "printlevel": 0
            },
            "nonlin-solver": {
                "type": "inexactnewton",
                "printlevel": 3,
                "maxiter": 50,
                "reltol": 1e-2,
                "abstol": 0.0,
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
                    "linear": False
                },
                "rotor": {
                    "attr": 5,
                    "material": "hiperco50",
                    "linear": False
                },
                "airgap": {
                    "attr": 6,
                    "material": "air",
                    "linear": True
                },
                "magnets": {
                    "material": "Nd2Fe14B",
                    "linear": True,
                    "attrs": list(range(7+4*num_slots, 7+4*num_slots+num_magnets))
                },
                "windings": {
                    "material": "copperwire",
                    "linear": True,
                    "attrs": list(range(7, 7+4*num_slots))
                }
            },
            "problem-opts": {
                "fill-factor": 1.0,
                "current-density": 1.0,
                "current" : {
                    "Phase-B":  [15, 16, 17, 18,
                            23, 24, 25, 26,
                            39, 40, 41, 42,
                            47, 48, 49, 50,
                            63, 64, 65, 66,
                            71, 72, 73, 74,
                            87, 88, 89, 90,
                            95, 96, 97, 98,
                    ],
                    "Phase-A":   [11, 12, 13, 14,
                            19, 20, 21, 22,
                            35, 36, 37, 38,
                            43, 44, 45, 46,
                            59, 60, 61, 62,
                            67, 68, 69, 70,
                            83, 84, 85, 86,
                            91, 92, 93, 94,
                    ],
                    # "off":  [7, 8, 9, 10,
                    #         27, 28, 29, 30,
                    #         31, 32, 33, 34,
                    #         51, 52, 53, 54,
                    #         55, 56, 57, 58,
                    #         75, 76, 77, 78,
                    #         79, 80, 81, 82,
                    #         99, 100, 101, 102,
                    # ]
                },
                "magnets": {
                    "north": north,
                    "cw": cw,
                    "south": south,
                    "ccw": ccw
                }
            },
            "bcs": {
                "essential": "all"
            }
        }

        solver = MachSolver("Magnetostatic", options)

        state = solver.getNewField()
        zero = Vector(np.array([0.0, 0.0, 0.0]))
        solver.setFieldValue(state, zero);

        current_density = 1.1e7 # 11 A/mm^2
        fill_factor = 1.0
        inputs = {
            "current-density": current_density,
            "fill-factor": fill_factor,
            "state": state
        }
        solver.solveForState(inputs, state)

        B = solver.getField("B")
        solver.printField("B", B, "B", 0, rotation)

        torque_options = {
            "attributes": [5] + magnets,
            "axis": [0.0, 0.0, 1.0],
            "about": [0.0, 0.0, 0.0]
        }
        solver.createOutput("torque", torque_options);

        torque.append(solver.calcOutput("torque", inputs))
        print(torque)

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



