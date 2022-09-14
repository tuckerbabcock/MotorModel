from pathlib import Path

from motormodel import Motor

_mesh_file = "pw127_e.smb"
_egads_file = "pw127_e.egads"
_csm_file = "pw127_e.csm"

_mesh_path = Path(__file__).parent / "mesh" / _mesh_file
_egads_path = Path(__file__).parent / "mesh" / _egads_file
_csm_path = Path(__file__).parent / "model" / _csm_file

_stator_attrs = [1, 3, 8, 30, 29, 26, 25, 22, 21, 18, 17, 14, 13, 10, 109, 106, 105, 102, 101, 98, 97, 94, 93, 90, 89, 86, 85, 82, 81, 78, 77, 74, 73, 70, 69, 66, 65, 62, 61, 58, 57, 54, 53, 50, 49, 46, 45, 42, 41, 38, 37, 34, 6, 2]
_rotor_attrs = [198]
_airgap_attrs = [5]
_current_attrs = [4, 9, 32, 31, 28, 27, 24, 23, 20, 19, 16, 15, 12, 11, 108, 107, 104, 103, 100, 99, 96, 95, 92, 91, 88, 87, 84, 83, 80, 79, 76, 75, 72, 71, 68, 67, 64, 63, 60, 59, 56, 55, 52, 51, 48, 47, 44, 43, 40, 39, 36, 35, 33, 7]
_magnet_attrs = [143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142]
# _magnet_attrs = [176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175]
_heatsink_attrs = []
_shaft_attrs = []

# _magnet_divisions = 1
# _rotations = [0]
# _multipoint_opts = _buildMultipointOptions(_magnet_attrs, _magnet_divisions, _rotations)

_current_indices = {
    "phaseA": {
        "z": [8, 13, 18, 23, 33, 42, 43, 52, 53],
        "-z": [1, 10, 11, 20, 21, 30, 40, 45, 50]
        # "z": [10, 13, 20, 23, 33, 43, 44, 53, 54],
        # "-z": [1, 11, 12, 21, 22, 32, 42, 45, 52]
    },
    "phaseB": {
        "z": [0, 5, 15, 24, 25, 34, 35, 44, 49],
        "-z": [2, 3, 12, 22, 27, 32, 37, 46, 47]
        # "z": [2, 5, 15, 25, 26, 35, 36, 46, 49],
        # "-z": [3, 4, 14, 24, 27, 34, 37, 47, 48]
    },
    "phaseC": {
        "z": [6, 7, 16, 17, 26, 31, 36, 41, 51],
        "-z": [4, 9, 14, 19, 28, 29, 38, 39, 48]
        # "z": [7, 8, 17, 18, 28, 31, 38, 41, 51],
        # "-z": [6, 9, 16, 19, 29, 30, 39, 40, 50]
    }
}

hiperco_reluctivity = {
    "model": "lognu",
    "cps": [5.5286, 5.4645, 4.5597, 4.2891, 3.8445, 4.2880, 4.9505, 11.9364, 11.9738, 12.6554, 12.8097, 13.3347, 13.5871, 13.5871, 13.5871],
    "knots": [0, 0, 0, 0, 0.1479, 0.5757, 0.9924, 1.4090, 1.8257, 2.2424, 2.6590, 3.0757, 3.4924, 3.9114, 8.0039, 10.0000, 10.0000, 10.0000, 10.0000],
    "degree": 3
}

_components = {
    "stator": {
        "attrs": _stator_attrs,
        "material": {
            "name": "hiperco50",
            "reluctivity": hiperco_reluctivity
        },
    },
    "rotor": {
        "attrs": _rotor_attrs,
        "material": {
            "name": "hiperco50",
            "reluctivity": hiperco_reluctivity
        },
    },
    "airgap": {
        "attrs": _airgap_attrs,
        "material": "air"
    },
    "shaft": {
        "attrs": _shaft_attrs,
        "material": "air"
    },
    "heatsink": {
        "attrs": _heatsink_attrs,
        "material": "air"
    },
    "windings": {
        "attrs": _current_attrs,
        "material": "copperwire"
    },
    "magnets": {
        "attrs": _magnet_attrs,
        "material": "Nd2Fe14B"
    }
}

# _multipoint_rotations = [0, 1, 2, 3]
_multipoint_rotations = [0]
_hallbach_segments = 4

class PW127E(Motor):
    def __init__(self, **kwargs):
        super().__init__(components=_components,
                         current_indices=_current_indices,
                         mesh_path=_mesh_path,
                         egads_path=_egads_path,
                         csm_path=_csm_path,
                         multipoint_rotations=_multipoint_rotations,
                         hallbach_segments=_hallbach_segments,
                         **kwargs)