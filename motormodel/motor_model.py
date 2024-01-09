import openmdao.api as om
from mphys import Multipoint
from omESP import omESP
# from invertermodel import Inverter

from mach import MachBuilder

from .scenario_motor import ScenarioMotor
from .motor_em_builder import EMMotorBuilder
from .motor_options import _buildSolverOptions
from .valid_geometry import ValidLengths
from .internal_cooling import InternalCooling, AirgapCooling
# from .tms import ThermalManagementSystem

try:
    from collections.abc import Mapping
except ImportErrror:
    from collections import Mapping


def _nested_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.

    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = _nested_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


class Motor(Multipoint):
    def initialize(self):
        # Required options:
        self.options.declare("components", types=dict, desc="")
        self.options.declare("em_bcs", types=dict, desc="")
        self.options.declare("thermal_bcs", types=dict, desc="")
        self.options.declare("thermal_interfaces", types=dict, desc="")
        self.options.declare("current_indices", types=dict, desc="")
        self.options.declare("mesh_path", desc="")
        self.options.declare("egads_path", desc="")
        self.options.declare("csm_path", desc="")
        # Optional options:
        self.options.declare("multipoint_rotations",
                             types=list, desc="", default=[0])
        self.options.declare("hallbach_segments",
                             types=int, desc="", default=4)
        self.options.declare("theta_e_offset", default=0.0)
        self.options.declare("coupled", default=None)
        self.options.declare("em_options", types=dict, default=None)
        self.options.declare("thermal_options", types=dict, default=None)
        self.options.declare("warper_options", types=dict, default=None)
        self.options.declare("run_name")
        # self.options.declare("em_paraview_dir", types=str, default="motor_em")
        # self.options.declare("thermal_paraview_dir", types=str, default="motor_thermal")
        self.options.declare("geom_partials", desc="", default=None)
        self.options.declare("two_dimensional", types=bool,
                             default=True, desc=" Use a two dimensional FEA model")
        self.options.declare("check_partials", default=False)

    def setup(self):
        two_dimensional = self.options["two_dimensional"]

        esp_outputs = ["x_surf", "num_slots",
                       "num_magnets", "magnet_divisions"]
        if not two_dimensional:
            esp_outputs.append("model_depth")

        egads_path = self.options["egads_path"]
        csm_path = self.options["csm_path"]
        geom_partials = self.options["geom_partials"]
        self.add_subsystem("geom",
                           omESP(csm_file=str(csm_path),
                                 egads_file=str(egads_path),
                                 partials=geom_partials),
                           promotes_inputs=["*"],
                           promotes_outputs=esp_outputs)

        self.add_subsystem("valid_lengths",
                           ValidLengths(),
                           promotes=['*'])

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
        em_bcs = self.options["em_bcs"]
        thermal_bcs = self.options["thermal_bcs"]
        thermal_interfaces = self.options["thermal_interfaces"]
        multipoint_rotations = self.options["multipoint_rotations"]
        current_indices = self.options["current_indices"]
        theta_e_offset = self.options["theta_e_offset"]
        _warper_options, _em_options, _thermal_options = _buildSolverOptions(components,
                                                                             em_bcs,
                                                                             thermal_bcs,
                                                                             thermal_interfaces,
                                                                             multipoint_rotations,
                                                                             num_magnets,
                                                                             magnet_divisions,
                                                                             two_dimensional,
                                                                             hallbach_segments,
                                                                             theta_e_offset,
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
            # _warper_options.update(self.options["warper_options"])
            _nested_update(_warper_options, self.options["warper_options"])
        if self.options["em_options"] is not None:
            # _em_options.update(self.options["em_options"])
            _nested_update(_em_options, self.options["em_options"])
        if self.options["thermal_options"] is not None:
            # _thermal_options.update(self.options["thermal_options"])
            _nested_update(_thermal_options, self.options["thermal_options"])

        run_name = self.options["run_name"]
        if run_name is not None:
            _em_options["paraview"]["directory"] = f"{run_name}_motor_em"
            _thermal_options["paraview"]["directory"] = f"{run_name}_motor_thermal"
        else:
            _em_options["paraview"]["directory"] = "motor_em"
            _thermal_options["paraview"]["directory"] = "motor_thermal"

        check_partials = self.options["check_partials"]

        em_motor_builder = EMMotorBuilder(solver_options=_em_options,
                                          warper_type="MeshWarper",
                                          warper_options=_warper_options,
                                          coupled=self.options["coupled"],
                                          two_dimensional=two_dimensional,
                                          check_partials=check_partials)

        em_motor_builder.initialize(self.comm)

        coupled = self.options["coupled"]
        # If coupling to thermal solver, compute heat sources...
        if coupled == "thermal" or coupled == "thermal:feedforward":

            self.add_subsystem("internal_cooling_params",
                               InternalCooling(coolant_fluid="PGW30"),
                               promotes_inputs=[
                                   ("fluid_temp", "fluid_temp:in-slot-cooling"),
                                   ("duct_length", "stack_length"),
                                   ("duct_width", "coolant_thickness"),
                                   ("duct_height", "slot_depth"),
                                   ("fluid_velocity", "coolant_velocity"),
                                   ("num_ducts", "num_slots")
                               ],
                               promotes_outputs=[
                                   ("heat_transfer_coefficient",
                                    "h:in-slot-cooling"),
                                   "flow_loss"
                               ])

            self.add_subsystem("airgap_cooling_params",
                               AirgapCooling(airgap_fluid="air"),
                               promotes_inputs=[
                                   ("fluid_temp", "fluid_temp:airgap-convection"),
                                   "rotor_or",
                                   "stator_ir",
                                   "rpm",
                               ],
                               promotes_outputs=[
                                   ("heat_transfer_coefficient",
                                    "h:airgap-convection"),
                               ])

            # thermal_flux_attributes = _thermal_options["bcs"]["flux"] # TODO: Answer QUESTION: what to put for thermal_flux boundary attributes? Attributes are in the other solver I believe
            # thermal_flux_attributes = _thermal_options["bcs"]["essential"] # TODO: Answer QUESTION: what to put for thermal_flux boundary attributes? Attributes are in the other solver I believe
            thermal_flux_depends = ["state",
                                    "mesh_coords"]  # TODO: Answer QUESTION: what to put for thermal_flux depends? Should just be able to list things out

            thermal_outputs = {
                # "thermal_flux" : {
                #     "options" : {
                #         "attributes" : _thermal_options["bcs"]["convection"]
                #     },
                #     "depends" : ["state", "mesh_coords"]
                # },
                # "max_state": {
                #     "options": {
                #         "rho": 10.0
                #     },
                #     "depends": ['state', 'mesh_coords']
                # },
                # "average_state": {
                #     "depends": ['state', 'mesh_coords']
                # },
                ("max_winding_temperature", "max_state:windings"): {
                    "options": {
                        "rho": 10.0,
                        "attributes": _thermal_options["components"]["windings"]["attrs"]
                    },
                    "depends": ['state', 'mesh_coords']
                },
                ("max_magnet_temperature", "max_state:magnets"): {
                    "options": {
                        "rho": 1.0,
                        "attributes": _thermal_options["components"]["magnets"]["attrs"]
                    },
                    "depends": ['state', 'mesh_coords']
                }
            }

            # print(f"max winding temp attrs: {_thermal_options['components']['windings']['attrs']}")

            thermal_builder = MachBuilder(solver_type="thermal",
                                          solver_options=_thermal_options,
                                          solver_inputs=[
                                              "h",
                                              "fluid_temp",
                                              "thermal_load",
                                              "fill_factor",
                                              "h:in-slot-cooling",
                                              "fluid_temp:in-slot-cooling",
                                              "h:airgap-convection"
                                          ],
                                          warper_type=None,
                                          warper_options=_warper_options,
                                          outputs=thermal_outputs,
                                          check_partials=check_partials)
            thermal_builder.initialize(self.comm)
        else:
            thermal_builder = None

        self.mphys_add_scenario("motor",
                                ScenarioMotor(em_motor_builder=em_motor_builder,
                                              thermal_builder=thermal_builder))

        # self.add_subsystem("inverter",
        #                    Inverter(),
        #                    promotes_inputs=[('I_phase_rms', 'rms_current'),
        #                                     ('r_wire', 'strand_radius'),
        #                                     ('load_inductance', 'L'),
        #                                     ('load_phase_back_emf',
        #                                      'phase_back_emf'),
        #                                     ('load_phase_resistance',
        #                                      'stator_phase_resistance'),
        #                                     ('electrical_frequency', 'frequency'),
        #                                     'bus_voltage'])

        # self.add_subsystem("total_loss",
        #                    om.ExecComp(
        #                        "total_loss = motor_loss + inverter_loss"),
        #                    promotes=[("inverter_loss", "inverter.total_loss"),
        #                              ("motor_loss", "motor.total_loss")])

        # self.add_subsystem("tms",
        #                    ThermalManagementSystem(),
        #                    promotes_inputs=["total_loss"],
        #                    promotes_outputs=["tms_mass", "tms_power_req"])

        # self.add_subsystem("total_mass",
        #                    om.ExecComp("total_mass = motor_mass + tms_mass"),
        #                    promotes=['*'])
        # self.add_subsystem("total_power_in",
        #                    om.ExecComp(
        #                        "total_power_in = motor_power_in + tms_power_req + flow_loss + inverter_loss"),
        #                    promotes_inputs=[("motor_power_in", "motor.power_in"),
        #                                     "tms_power_req",
        #                                     "flow_loss",
        #                                     ("inverter_loss", "inverter.loss")],
        #                    promotes_outputs=['total_power_in'])

        # self.add_subsystem("efficiency",
        #                    om.ExecComp(
        #                        "efficiency = motor_power_out / total_power_in"),
        #                    promotes_inputs=[("motor_power_out", "motor.power_out"),
        #                                     "total_power_in"],
        #                    promotes_outputs=['efficiency'])

        # self.connect("x_surf", "x_em")
        # self.connect("x_surf", "x_em_vol", src_indices=[i for i in range(9) if i % 3 != 0 ])
        # self.connect("x_surf", "x_em_vol")

        # TODO: Determine if this is OK
        # Could not find any instances of "x_conduct" in MotorModel files or mach repo. Was causing error by virtue of it not existing, so commented out.
        # if coupled == "thermal":
        #     self.connect("x_surf", "x_conduct")

        motor_promotes = [
            "average_torque",
            #    "energy",
            "ac_loss",
            "dc_loss",
            "stator_core_loss",
            "total_loss",
            "motor_mass",
            #    "max_flux_magnitude:stator",
            #    "max_flux_magnitude:winding",
            #    "stator_volume",
            "average_flux_magnitude:airgap",
            #    "winding_max_peak_flux",
            "rms_current",
            # "efficiency",
            "power_out",
            "power_in",
            "fill_factor",
            "phase_back_emf",
            "stator_phase_resistance",
            'L'
        ]

        if thermal_builder is not None:
            for output in thermal_outputs:
                if isinstance(output, str):
                    motor_promotes.append(output)
                elif isinstance(output, tuple):
                    motor_promotes.append(output[0])

        self.promotes("motor",
                      inputs=["*"],
                      outputs=motor_promotes)

    def configure(self):
        two_dimensional = self.options["two_dimensional"]
        coupled = self.options["coupled"]
        if two_dimensional:
            x_surf_metadata = self.geom.get_io_metadata(
                'output', metadata_keys=['size'])['x_surf']
            print("x_surf_metadata: ", x_surf_metadata)
            x_surf_size = x_surf_metadata['size']
            self.connect("x_surf", "x_em_vol", src_indices=[
                         i for i in range(x_surf_size) if (i+1) % 3 != 0])
            if coupled == "thermal" or coupled == "thermal:feedforward":
                self.connect("x_surf", "x_conduct_vol", src_indices=[
                             i for i in range(x_surf_size) if (i+1) % 3 != 0])
        else:
            self.connect("x_surf", "x_em")

        geom_inputs = self.geom.get_io_metadata('input', metadata_keys=['val'])

        for geom_input in geom_inputs:
            self.set_input_defaults(geom_input, geom_inputs[geom_input]['val'])

        geom_outputs = self.geom.get_io_metadata(
            'output', metadata_keys=['val'])
        self.set_input_defaults("num_slots", geom_outputs["num_slots"]['val'])

        # self.set_input_defaults("tooth_tip_thickness", inputs['tooth_tip_thickness']['val'])
        # self.set_input_defaults("tooth_tip_angle", inputs['tooth_tip_angle']['val'])
        # self.set_input_defaults("slot_radius", inputs['slot_radius']['val'])
        # self.set_input_defaults("shoe_spacing", inputs['shoe_spacing']['val'])
        # self.set_input_defaults("slot_depth", inputs['slot_depth']['val'])
        # self.set_input_defaults("tooth_width", inputs['tooth_width']['val'])
        # self.set_input_defaults("stator_id", inputs['stator_id']['val'])
        # # self.set_input_defaults("stack_length", inputs['shoe_spacing']['val'])

        self.set_input_defaults('slot_depth', units='m')
        self.set_input_defaults('coolant_thickness', units='m')
        self.set_input_defaults(
            'fluid_temp:in-slot-cooling', units='K', val=300)
        self.set_input_defaults('rpm', units='rpm', val=1000)

        self.set_input_defaults('stack_length', units='m', val=1.0)
        self.set_input_defaults('strand_radius', units='m', val=1.0)
