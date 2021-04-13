import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mach import MachSolver, Mesh, Vector

options = {
    "silent": False,
    "print-options": False,
    "mesh": {
        "file": "mesh/motor2D.smb",
        "model-file": "mesh/motor2D.egads"
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
        "airgap": {
            "attr": 3,
            "material": "air",
            "linear": True
        },
        "magnets": {
            "material": "Nd2Fe14B",
            "linear": True,
            "attrs": [4, 5, 6, 7]
        },
        "windings": {
            "material": "copperwire",
            "linear": True,
            "attrs": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        }
    },
    "problem-opts": {
        "fill-factor": 1.0,
        "current-density": 1.0,
        "current" : {
            "Phase-A": [8, 9, 14, 15],
            "Phase-B": [10, 11, 16, 17],
            "Phase-C": [12, 13, 18, 19]
        },
        "magnets": {
            "south": [4, 6],
            "north": [5, 7],
        }
    },
    "bcs": {
        "essential": "all"
    }
}

if __name__ == "__main__":
    solver = MachSolver("Magnetostatic", options)

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
        "attributes": [2, 4, 5, 6, 7],
        "axis": [0.0, 0.0, 1.0],
        "about": [0.0, 0.0, 0.0]
    }
    solver.createOutput("torque", torque_options);

    torque = solver.calcOutput("torque", inputs)
    print("Torque: ", torque)



