import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mach import MachSolver, Mesh, Vector

num_magnets_true = 40
num_magnets = 40
mag_pitch = num_magnets // num_magnets_true
num_slots = 24

start = 0
nturns = 4
torque = []

if __name__ == "__main__":
    for rotation in range(start, start+nturns):
    # for rotation in range(nturns, 2*nturns):
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
                "type": "hypregmres",
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
                "type": "relaxed-newton",
                "printlevel": 3,
                "maxiter": 20,
                "reltol": 1e-4,
                "abstol": 5e-1,
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
                "fill-factor": 1.0,
                "current-density": 1.0,
                "current" : {
                    "z": [43, 46, 47, 50, 7, 10, 11, 14, 19, 22, 23, 26, 31, 34, 35, 38],
                    "-z": [44, 45, 48, 49, 8, 9, 12, 13, 20, 21, 24, 25, 32, 33, 36, 37]
                    # "z": [9, 10, 15, 16],
                    # "-z": [5, 8, 11, 14]
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


        current_density = 11e6 # 11 A/mm^2
        fill_factor = 0.8
        inputs = {
            "current-density": current_density,
            "fill-factor": fill_factor,
            "state": state
        }
        solver.solveForState(inputs, state)

        B = solver.getField("B")
        solver.printField("B", B, "B", 0, rotation)

        torque_options = {
            "attributes": [2] + magnets,
            "axis": [0.0, 0.0, 1.0],
            "about": [0.0, 0.0, 0.0]
        }
        solver.createOutput("torque", torque_options);

        torque.append(solver.calcOutput("torque", inputs))
        print(torque)

    print("Torque: ", torque)



