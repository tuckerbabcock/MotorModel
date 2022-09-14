import os
from pathlib import Path

import numpy as np

import openmdao.api as om
from mphys import Multipoint
from omESP import omESP

from mach import MachBuilder

from .scenario_motor import ScenarioMotor
from .motor_em_builder import EMMotorBuilder
from .motor_options import _buildSolverOptions

# _mesh_file = "mesh_motor2D.smb"
# _egads_file = "mesh_motor2D.egads"
# _csm_file = "motor2D.csm"

# _2D_mesh_file = "mesh_motor2D_true.smb"
# _2D_egads_file = "mesh_motor2D_true.egads"
# _2D_csm_file = "motor2D_true.csm"

# _2D_mesh_file = "pw127_e.smb"
# _2D_egads_file = "pw127_e.egads"
# _2D_csm_file = "pw127_e.csm"

class Motor(Multipoint): 
    def initialize(self):
        # Required options:
        self.options.declare("components", types=dict, desc="")
        self.options.declare("current_indices", types=dict, desc="")
        self.options.declare("mesh_path", desc="")
        self.options.declare("egads_path", desc="")
        self.options.declare("csm_path", desc="")
        # Optional options:
        self.options.declare("multipoint_rotations", types=list, desc="", default=[0])
        self.options.declare("hallbach_segments", types=int, desc="", default=4)
        self.options.declare("coupled", default=None)
        self.options.declare("em_options", types=dict, default=None)
        self.options.declare("thermal_options", types=dict, default=None)
        self.options.declare("warper_options", types=dict, default=None)
        self.options.declare("two_dimensional", types=bool, default=True, desc=" Use a two dimensional FEA model")
        self.options.declare("check_partials", default=False)

    def setup(self):
        two_dimensional = self.options["two_dimensional"]

        esp_outputs = ["x_surf", "num_slots", "num_magnets", "magnet_divisions"]
        if not two_dimensional:
            esp_outputs.append("model_depth")

        egads_path = self.options["egads_path"]
        csm_path = self.options["csm_path"]
        self.add_subsystem("geom",
                           omESP(csm_file=str(csm_path),
                                 egads_file=str(egads_path)),
                           promotes_inputs=["*"],
                           promotes_outputs=esp_outputs)

        geom_config_values = self.geom.getConfigurationValues()
        num_magnets = int(geom_config_values["num_magnets"])
        magnet_divisions = int(geom_config_values["magnet_divisions"])
        num_slots = int(geom_config_values["num_slots"])

        self.add_subsystem("convert", om.ExecComp("stator_inner_radius = stator_id / 2"),
                            promotes=["*"])

        hallbach_segments = self.options["hallbach_segments"]
        num_poles = num_magnets / (hallbach_segments / 2)
        self.add_subsystem("frequency",
                           om.ExecComp(f"frequency = rpm * {num_poles} / 120"),
                           promotes=["*"])

        components = self.options["components"]
        multipoint_rotations = self.options["multipoint_rotations"]
        current_indices = self.options["current_indices"]
        _warper_options, _em_options, _thermal_options = _buildSolverOptions(components,
                                                                             multipoint_rotations,
                                                                             magnet_divisions, 
                                                                             two_dimensional,
                                                                             hallbach_segments,
                                                                             current_indices)

        mesh_path = self.options["mesh_path"]
        _warper_options["mesh"] = {}
        _warper_options["mesh"]["file"] = str(mesh_path)
        _warper_options["mesh"]["model-file"] = str(egads_path)
        _em_options["mesh"] = {}
        _em_options["mesh"]["file"] = str(mesh_path)
        _em_options["mesh"]["model-file"] = str(egads_path)
        _thermal_options["mesh"] = {}
        _thermal_options["mesh"]["file"] = str(mesh_path)
        _thermal_options["mesh"]["model-file"] = str(egads_path)

        if self.options["warper_options"] is not None:
            _warper_options.update(self.options["warper_options"])
        if self.options["em_options"] is not None:
            _em_options.update(self.options["em_options"])
        if self.options["thermal_options"] is not None:
            _thermal_options.update(self.options["thermal_options"])

        check_partials=self.options["check_partials"]

        em_motor_builder = EMMotorBuilder(solver_options=_em_options,
                                          warper_type="MeshWarper",
                                          warper_options=_warper_options,
                                          coupled=self.options["coupled"],
                                          two_dimensional=two_dimensional,
                                          check_partials=check_partials)

        em_motor_builder.initialize(self.comm)

        coupled = self.options["coupled"]
        # If coupling to thermal solver, compute heat sources...
        if coupled == "thermal":
            thermal_builder = MachBuilder(solver_type="thermal",
                                          solver_options=_thermal_options,
                                          solver_inputs=["h", "fluid_temp", "thermal_load"],
                                          warper_type="MeshWarper",
                                          warper_options=warper_options,
                                          outputs={},
                                          check_partials=check_partials)
            thermal_builder.initialize(self.comm)
        else:
            thermal_builder = None

        self.mphys_add_scenario("analysis",
                                ScenarioMotor(em_motor_builder=em_motor_builder,
                                              thermal_builder=thermal_builder))

        # self.connect("x_surf", "x_em")
        # self.connect("x_surf", "x_em_vol", src_indices=[i for i in range(9) if i % 3 != 0 ])
        # self.connect("x_surf", "x_em_vol")

        if coupled == "thermal":
            self.connect("x_surf", "x_conduct")

        self.promotes("analysis",
                      inputs=["*"], 
                      outputs=[
                               "average_torque",
                               "energy",
                               "ac_loss",
                               "dc_loss",
                               "stator_core_loss",
                               "stator_mass",
                               "max_flux_magnitude:stator",
                            #    "max_flux_magnitude:winding",
                               "stator_volume",
                               "average_flux_magnitude:airgap",
                            #    "winding_max_peak_flux",
                               "rms_current",
                               "efficiency",
                               "power_out",
                               "power_in"
                               ]
                     )

    def configure(self):
        two_dimensional = self.options["two_dimensional"]
        if two_dimensional:
            x_surf_metadata = self.geom.get_io_metadata('output', metadata_keys=['size'])['x_surf']
            print("x_surf_metadata: ", x_surf_metadata)
            x_surf_size = x_surf_metadata['size']
            self.connect("x_surf", "x_em_vol", src_indices=[i for i in range(x_surf_size) if (i+1) % 3 != 0 ])
        else:
            self.connect("x_surf", "x_em")

        geom_inputs = self.geom.get_io_metadata('input', metadata_keys=['val'])

        for geom_input in geom_inputs:
            self.set_input_defaults(geom_input, geom_inputs[geom_input]['val'])

        # self.set_input_defaults("tooth_tip_thickness", inputs['tooth_tip_thickness']['val'])
        # self.set_input_defaults("tooth_tip_angle", inputs['tooth_tip_angle']['val'])
        # self.set_input_defaults("slot_radius", inputs['slot_radius']['val'])
        # self.set_input_defaults("shoe_spacing", inputs['shoe_spacing']['val'])
        # self.set_input_defaults("slot_depth", inputs['slot_depth']['val'])
        # self.set_input_defaults("tooth_width", inputs['tooth_width']['val'])
        # self.set_input_defaults("stator_id", inputs['stator_id']['val'])
        # # self.set_input_defaults("stack_length", inputs['shoe_spacing']['val'])
    