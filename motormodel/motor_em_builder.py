import copy

import openmdao.api as om
from mphys import Builder

from mach import PDESolver, MeshWarper
from mach import MachState, MachMeshWarper, MachFunctional, MachMeshGroup

from .average_comp import AverageComp
from .maximum_fit import DiscreteInducedExponential
from .motor_current import MotorCurrent
from .dc_loss import WireLength, DCLoss
from .parks_transform import ParksTransform
from .inductance import Inductance


class EMStateAndFluxMagGroup(om.Group):
    def initialize(self):
        self.options.declare("solver", types=PDESolver, recordable=False)
        self.options.declare("state_depends", types=list)
        self.options.declare("check_partials", default=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solver = self.options["solver"]
        depends = self.options["state_depends"]
        self.check_partials = self.options["check_partials"]

        self.add_subsystem("state",
                           MachState(solver=self.solver,
                                     depends=depends,
                                     check_partials=self.check_partials),
                           promotes_inputs=[
                               ("mesh_coords", "x_em_vol"), *depends[1:]],
                           promotes_outputs=[("state", "em_state")])

        self.add_subsystem("flux_magnitude",
                           MachFunctional(solver=self.solver,
                                          func="flux_magnitude",
                                          depends=["state", "mesh_coords"],
                                          check_partials=self.check_partials),
                           promotes_inputs=[
                               ("state", "em_state"), ("mesh_coords", "x_em_vol")],
                           promotes_outputs=["flux_magnitude"])

        # # Flux density used for demagnetization proximity constraint
        # self.add_subsystem("flux_density",
        #                    MachFunctional(solver=self.solver,
        #                                   func="flux_density",
        #                                   depends=["state", "mesh_coords"],
        #                                   check_partials=self.check_partials),
        #                    promotes_inputs=[("state", "em_state"), ("mesh_coords", "x_em_vol")],
        #                    promotes_outputs=["flux_density"])


class EMMotorCouplingGroup(om.Group):
    def initialize(self):
        self.options.declare("solvers", types=list, recordable=False)
        self.options.declare("state_depends", types=list)
        self.options.declare("coupled", default=False)
        self.options.declare("check_partials", default=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solvers = self.options["solvers"]
        depends = self.options["state_depends"]
        self.check_partials = self.options["check_partials"]
        coupled = self.options["coupled"]

        if coupled != "thermal":
            temperature_name = "reference_temperature"
        else:
            temperature_name = "temperature"

        # em_states = self.add_subsystem("em_states", om.ParallelGroup())
        em_states = self.add_subsystem("em_states", om.Group())
        for idx, solver in enumerate(self.solvers):
            em_states.add_subsystem(f"solver{idx}",
                                    EMStateAndFluxMagGroup(solver=solver,
                                                           state_depends=depends,
                                                           check_partials=self.check_partials))

            self.promotes("em_states",
                          inputs=[(f"solver{idx}.x_em_vol", "x_em_vol"),
                                  (f"solver{idx}.temperature",
                                   temperature_name),
                                  *[f"solver{idx}.{input}" for input in depends[2:]]],
                          outputs=[(f"solver{idx}.em_state", f"em_state{idx}")])

        self.add_subsystem("peak_flux",
                           DiscreteInducedExponential(num_pts=len(self.solvers),
                                                      rho=10),
                           promotes_outputs=[("data_amplitude", "peak_flux")])

        for idx, _ in enumerate(self.solvers):
            self.connect(
                f"em_states.solver{idx}.flux_magnitude", f"peak_flux.data{idx}")

        # If coupling to thermal solver, compute heat sources...
        if coupled == "thermal" or coupled == "thermal:feedforward":
            # self.add_subsystem("stator_max_flux_magnitude",
            #                    MachFunctional(solver=self.solvers[0],
            #                                   func="max_flux_magnitude:stator",
            #                                   func_options={"rho": 1, "attributes": [1]},
            #                                   depends=["state", "mesh_coords"]),
            #                    promotes_inputs=[("mesh_coords", "x_em_vol"),
            #                                     ("state", "em_state0")],
            #                    promotes_outputs=["max_flux_magnitude:stator"])

            # self.add_subsystem("stator_max_flux_magnitude",
            #                    MachFunctional(solver=self.solvers[0],
            #                                   func="max_state:stator",
            #                                   func_options={
            #                                     "rho": 10,
            #                                     "attributes": stator_attrs,
            #                                     "state": "peak_flux"
            #                                   },
            #                                   depends=["state", "mesh_coords"]),
            #                    promotes_inputs=[("mesh_coords", "x_em_vol"),
            #                                     ("state", "peak_flux")],
            #                    promotes_outputs=[("max_state:stator", "max_flux_magnitude:stator")])

            stator_attrs = self.solvers[0].getOptions(
            )["components"]["stator"]["attrs"]
            # rotor_attrs = self.solvers[0].getOptions()["components"]["rotor"]["attrs"]
            rotor_attrs = []
            winding_attrs = self.solvers[0].getOptions(
            )["components"]["windings"]["attrs"]

            heat_source_inputs = [("mesh_coords", "x_em_vol"),
                                  ("temperature", temperature_name),
                                  "frequency",
                                  #   "max_flux_magnitude:stator", # going to use CAL model for Aviation paper
                                  "wire_length",
                                  "rms_current",
                                  "strand_radius",
                                  "strands_in_hand",
                                  "stack_length",
                                  "peak_flux",
                                  "model_depth",
                                  "num_turns",
                                  "num_slots"]
            self.add_subsystem("heat_source",
                               MachFunctional(solver=self.solvers[0],
                                              func="heat_source",
                                              func_options={
                                                  "dc_loss": {
                                                      "attributes": winding_attrs
                                                  },
                                                  "ac_loss": {
                                                      "attributes": winding_attrs
                                                  },
                                                  "core_loss": {
                                                      "attributes": [*stator_attrs, *rotor_attrs]
                                                  },
                               },
                                   depends=heat_source_inputs,
                                   check_partials=self.check_partials),
                               promotes_inputs=heat_source_inputs,
                               promotes_outputs=[("heat_source", "thermal_load")])

        elif coupled is not None:
            raise ValueError(
                "EM Motor builder only supports coupling with a thermal solver")


class EMMotorPrecouplingGroup(om.Group):
    """
    Group that handles surface -> volume mesh movement

    To properly support parallel analysis, I'll need to have this component
    partition the input surface coords
    """

    def initialize(self):
        self.options.declare("solvers", types=list, recordable=False)
        self.options.declare("warper", recordable=False)
        self.options.declare("coupled", default=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solvers = self.options["solvers"]
        self.warper = self.options["warper"]
        coupled = self.options["coupled"]

        # # Promote variables with physics-specific tag that MPhys expects
        if isinstance(self.warper, MeshWarper):
            self.add_subsystem("mesh_warper",
                               MachMeshWarper(warper=self.warper),
                               promotes_inputs=[("surf_mesh_coords", "x_em")],
                               promotes_outputs=[("vol_mesh_coords", "x_em_vol")])

        theta_e = []
        for solver in self.solvers:
            solver_options = solver.getOptions()
            theta_e.append(solver_options["theta_e"])

        self.add_subsystem(f"current",
                           MotorCurrent(theta_e=theta_e),
                           promotes_inputs=["*"],
                           promotes_outputs=["rms_current",
                                             "current_density",
                                             "three_phase*.current_density:phase*",
                                             "three_phase*.current:phase*",
                                             "fill_factor"])

        # If coupling to thermal solver, compute wire length for heat sources to use
        if coupled == "thermal" or coupled == "thermal:feedforward":
            self.add_subsystem("wire_length",
                               WireLength(),
                               promotes_inputs=["*"],
                               promotes_outputs=["wire_length"])


class EMMotorOutputsGroup(om.Group):
    """
    Group that handles calculating outputs after the state solve
    """

    def initialize(self):
        self.options.declare("solvers", types=list, recordable=False)
        self.options.declare("coupled", default=False)
        self.options.declare("check_partials", default=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solvers = self.options["solvers"]
        self.check_partials = self.options["check_partials"]

        coupled = self.options["coupled"]

        if coupled != "thermal":
            temperature_name = "reference_temperature"
        else:
            temperature_name = "temperature"

        # em_functionals = self.add_subsystem("em_functionals", om.ParallelGroup())
        torque = self.add_subsystem("torque", om.Group())
        for idx, solver in enumerate(self.solvers):

            rotor_attrs = solver.getOptions()["components"]["rotor"]["attrs"]
            magnet_attrs = solver.getOptions(
            )["components"]["magnets"]["attrs"]
            airgap_attrs = solver.getOptions()["components"]["airgap"]["attrs"]
            torque_opts = {
                "attributes": [*rotor_attrs, *magnet_attrs],
                "axis": [0.0, 0.0, 1.0],
                "about": [0.0, 0.0, 0.0],
                "air_attributes": airgap_attrs
            }

            torque.add_subsystem(f"torque{idx}",
                                 MachFunctional(solver=solver,
                                                func="torque",
                                                func_options=torque_opts,
                                                depends=[
                                                    "state", "mesh_coords"],
                                                check_partials=self.check_partials))

            self.promotes("torque",
                          inputs=[(f"torque{idx}.mesh_coords", "x_em_vol"),
                                  (f"torque{idx}.state", f"em_state{idx}")])

        self.add_subsystem("raw_avg_torque",
                           AverageComp(num_pts=len(self.solvers)),
                           promotes_outputs=[("data_average", "raw_average_torque")])

        for idx, _ in enumerate(self.solvers):
            self.connect(f"torque.torque{idx}.torque",
                         f"raw_avg_torque.data{idx}")

        self.add_subsystem("avg_torque",
                           om.ExecComp(
                               "average_torque = raw_average_torque * stack_length / model_depth"),
                           promotes=["*"])

        # flux_linkage = self.add_subsystem("flux_linkage", om.Group())
        # for idx, solver in enumerate(self.solvers):
        #     current_opts = solver.getOptions()["current"]
        #     for current_group, sources in current_opts.items():
        #         current_group_attrs = []
        #         for source, attrs in sources.items():
        #             current_group_attrs.extend(attrs)
        #             # if source == "z":
        #             #     source_name = "z"
        #             # else:
        #             #     source_name = "minus_z"

        #         flux_linkage_opts = {
        #             "attributes": current_group_attrs,
        #             # "attributes": attrs,
        #             "name": current_group
        #         }
        #         flux_linkage.add_subsystem(f"flux_linkage_{current_group}{idx}",
        #                                    MachFunctional(solver=solver,
        #                                                   func=f"flux_linkage:{current_group}",
        #                                                   func_options=flux_linkage_opts,
        #                                                   depends=[
        #                                                       "state", "mesh_coords", f"current:{current_group}"],
        #                                                   check_partials=self.check_partials))

        #         self.promotes("flux_linkage",
        #                       inputs=[(f"flux_linkage_{current_group}{idx}.mesh_coords", "x_em_vol"),
        #                               (f"flux_linkage_{current_group}{idx}.state", f"em_state{idx}")])

        # flux_linkage2 = self.add_subsystem("flux_linkage2", om.Group())
        # for idx, solver in enumerate(self.solvers):
        #     current_opts = solver.getOptions()["current"]
        #     for current_group, sources in current_opts.items():
        #         for source, attrs in sources.items():
        #             if source == "z":
        #                 source_name = "z"
        #             else:
        #                 source_name = "minus_z"
        #                 # continue

        #             flux_linkage_opts = {
        #                 "attributes": attrs,
        #             }
        #             flux_linkage2.add_subsystem(f"flux_linkage2_{current_group}{source_name}{idx}",
        #                                         MachFunctional(solver=solver,
        #                                                        func=f"flux_flux_linkage:{current_group}{source_name}",
        #                                                        func_options=flux_linkage_opts,
        #                                                        depends=[
        #                                                            "state", "mesh_coords"],
        #                                                        check_partials=self.check_partials))

        #             self.promotes("flux_linkage2",
        #                           inputs=[(f"flux_linkage2_{current_group}{source_name}{idx}.mesh_coords", "x_em_vol"),
        #                                   (f"flux_linkage2_{current_group}{source_name}{idx}.state", f"em_state{idx}")])

        # flux_linkage3 = self.add_subsystem("flux_linkage3", om.Group())
        # for idx, solver in enumerate(self.solvers):
        #     current_opts = solver.getOptions()["current"]
        #     for current_group, sources in current_opts.items():
        #         current_group_attrs = []
        #         for source, attrs in sources.items():
        #             current_group_attrs.extend(attrs)
        #             # if source == "z":
        #             #     source_name = "z"
        #             # else:
        #             #     source_name = "minus_z"

        #         flux_linkage_opts = {
        #             "attributes": current_group_attrs,
        #             # "attributes": attrs,
        #             # "name": current_group
        #         }
        #         flux_linkage3.add_subsystem(f"flux_linkage3_{current_group}{idx}",
        #                                     MachFunctional(solver=solver,
        #                                                    func=f"flux_flux_flux_linkage:{current_group}",
        #                                                    func_options=flux_linkage_opts,
        #                                                    depends=[
        #                                                        "state", "mesh_coords"],
        #                                                    check_partials=self.check_partials))

        #         self.promotes("flux_linkage3",
        #                       inputs=[(f"flux_linkage3_{current_group}{idx}.mesh_coords", "x_em_vol"),
        #                               (f"flux_linkage3_{current_group}{idx}.state", f"em_state{idx}")])

        flux_linkage = self.add_subsystem("flux_linkage", om.Group())
        for idx, solver in enumerate(self.solvers):
            solver_options = solver.getOptions()
            current_opts = solver_options["current"]
            for current_group, sources in current_opts.items():
                current_group_attrs = []
                output_names = []
                for source, attrs in sources.items():
                    current_group_attrs.extend(attrs)
                    if source == "-z":
                        source_name = "minus_z"
                    else:
                        source_name = source

                    flux_linkage_opts = {
                        "attributes": attrs,
                    }
                    flux_linkage.add_subsystem(f"flux_linkage{idx}_{current_group}_{source_name}",
                                               MachFunctional(solver=solver,
                                                              func=f"flux_linkage:{current_group}_{source_name}",
                                                              func_options=flux_linkage_opts,
                                                              depends=[
                                                                   "state", "mesh_coords"],
                                                              check_partials=self.check_partials),
                                               promotes_outputs=[(f"flux_linkage:{current_group}_{source_name}", f"flux_linkage{idx}_{current_group}_{source_name}")])

                    output_names.append(
                        f"flux_linkage{idx}_{current_group}_{source_name}")
                    self.promotes("flux_linkage",
                                  inputs=[(f"flux_linkage{idx}_{current_group}_{source_name}.mesh_coords", "x_em_vol"),
                                          (f"flux_linkage{idx}_{current_group}_{source_name}.state", f"em_state{idx}"),
                                          "stack_length"])

                exec_comp_string = f"flux_linkage{idx}_{current_group} = stack_length * ({output_names[0]}"
                for name in output_names[1:]:
                    exec_comp_string += f"+ {name}"
                exec_comp_string += ")"
                flux_linkage.add_subsystem(f"flux_linkage{idx}_{current_group}",
                                           om.ExecComp(exec_comp_string),
                                           promotes=["*"])

            flux_linkage.add_subsystem(f"d_q_flux_linkage{idx}",
                                       ParksTransform(
                                           theta_e=solver_options["theta_e"]),
                                       promotes_inputs=[("phaseA", f"flux_linkage{idx}_phaseA"),
                                                        ("phaseB",
                                                         f"flux_linkage{idx}_phaseB"),
                                                        ("phaseC", f"flux_linkage{idx}_phaseC")])
            #    promotes_outputs=[("d", f"d_axis_flux_linkage{idx}"),
            #                      ("q", f"q_axis_flux_linkage{idx}")])

        self.add_subsystem("inductance",
                           Inductance(n=len(self.solvers)),
                           promotes_outputs=['L'])

        # self.promotes("flux_linkage", any=["stack_length"])

        # self.add_subsystem("avg_torque",
        #                    om.ExecComp(
        #                        "average_torque = raw_average_torque * stack_length / model_depth"),
        #                    promotes=["*"])

        # self.add_subsystem("raw_avg_flux_linkage",
        #                    AverageComp(num_pts=len(self.solvers)),
        #                    promotes_outputs=[("data_average", "raw_average_torque")])

        # for idx, _ in enumerate(self.solvers):
        #     self.connect(f"torque.torque{idx}.flux_linkage", f"raw_avg_flux_linkage.data{idx}")

        # self.add_subsystem("avg_torque",
        #                    om.ExecComp("average_torque = raw_average_torque * stack_length / model_depth"),
        #                    promotes=["*"])
        # airgap_attrs = solver.getOptions()["components"]["airgap"]["attrs"]
        # self.add_subsystem("raw_energy",
        #                    MachFunctional(solver=self.solvers[0],
        #                                   func="energy",
        #                                   func_options={"attributes": airgap_attrs},
        #                                   depends=["state", "mesh_coords"],
        #                                   check_partials=self.check_partials),
        #                    promotes_inputs=[("mesh_coords", "x_em_vol"),
        #                                     ("state", "em_state0")],
        #                    promotes_outputs=[("energy", "raw_energy")])

        # self.add_subsystem("mag_energy",
        #                    om.ExecComp("energy = raw_energy * stack_length / model_depth"),
        #                    promotes=["*"])

        # # winding_attrs = self.solvers[0].getOptions()["components"]["windings"]["attrs"]
        # # self.add_subsystem("winding_max_flux_magnitude",
        # #                    MachFunctional(solver=self.solvers[0],
        # #                                   func="max_flux_magnitude:winding",
        # #                                   func_options={"rho": 50, "attributes": winding_attrs},
        # #                                   depends=["state", "mesh_coords"]),
        # #                    promotes_inputs=[("mesh_coords", "x_em_vol"),
        # #                                     ("state", "em_state0")],
        # #                    promotes_outputs=["max_flux_magnitude:winding"])

        airgap_attrs = self.solvers[0].getOptions(
        )["components"]["airgap"]["attrs"]
        self.add_subsystem("airgap_average_flux_magnitude",
                           MachFunctional(solver=self.solvers[0],
                                          func="average_flux_magnitude:airgap",
                                          func_options={
                                              "attributes": airgap_attrs},
                                          depends=["state", "mesh_coords"],
                                          check_partials=self.check_partials),
                           promotes_inputs=[("mesh_coords", "x_em_vol"),
                                            ("state", "em_state0")],
                           promotes_outputs=["average_flux_magnitude:airgap"])

        ac_loss_depends = ["mesh_coords",
                           "temperature",
                           "stack_length",
                           "frequency",
                           "peak_flux",
                           "strand_radius",
                           "model_depth",
                           "strands_in_hand",
                           "num_turns",
                           "num_slots"]

        winding_attrs = self.solvers[0].getOptions(
        )["components"]["windings"]["attrs"]
        self.add_subsystem("ac_loss",
                           MachFunctional(solver=self.solvers[0],
                                          func="ac_loss",
                                          func_options={
                                              "attributes": winding_attrs},
                                          depends=ac_loss_depends,
                                          check_partials=self.check_partials),
                           promotes_inputs=[
                               ("mesh_coords", "x_em_vol"), ("temperature", temperature_name), *ac_loss_depends[2:]],
                           promotes_outputs=["ac_loss"])

        self.add_subsystem("dc_loss",
                           DCLoss(solver=self.solvers[0]),
                           promotes_inputs=["x_em_vol",
                                            "num_slots",
                                            "num_turns",
                                            "num_slots",
                                            "stator_inner_radius",
                                            "tooth_tip_thickness",
                                            "slot_depth",
                                            "tooth_width",
                                            "stack_length",
                                            "rms_current",
                                            "strand_radius",
                                            "strands_in_hand",
                                            ("temperature", temperature_name)],
                           promotes_outputs=["*"])

        self.add_subsystem("stator_phase_resistance",
                           om.ExecComp(
                               "stator_phase_resistance = (ac_loss + dc_loss) / (3 * rms_current**2)"),
                           promotes=['*'])

        # self.add_subsystem("winding_max_peak_flux",
        #                    MachFunctional(solver=self.solvers[0],
        #                                   func="max_state",
        #                                   func_options={
        #                                       "rho": 1,
        #                                       "attributes": winding_attrs,
        #                                       "state": "flux_magnitude"
        #                                   },
        #                                   depends=["mesh_coords", "state"],
        #                                   check_partials=self.check_partials),
        #                    promotes_inputs=[("mesh_coords", "x_em_vol"),
        #                                     ("state", "peak_flux")],
        #                    promotes_outputs=[("max_state", "winding_max_peak_flux")])

        # coupled = self.options["coupled"]
        # If not coupling to thermal solver, compute stator max flux here post coupling
        # if coupled != "thermal" and coupled != "thermal_full": # TODO: Change conditional logic to separate one way and fully coupled
        # stator_attrs = self.solvers[0].getOptions()["components"]["stator"]["attrs"]
        # self.add_subsystem("stator_max_flux_magnitude",
        #                     MachFunctional(solver=self.solvers[0],
        #                                    func="max_flux_magnitude:stator",
        #                                    func_options={"rho": 10, "attributes": stator_attrs},
        #                                    depends=["state", "mesh_coords"],
        #                                    check_partials=self.check_partials),
        #                     promotes_inputs=[("mesh_coords", "x_em_vol"),
        #                                      ("state", "em_state0")],
        #                     promotes_outputs=["max_flux_magnitude:stator"])
        # self.add_subsystem("stator_max_flux_magnitude",
        #                    MachFunctional(solver=self.solvers[0],
        #                                   func="max_state:stator",
        #                                   func_options={
        #                                     "rho": 10,
        #                                     "attributes": stator_attrs,
        #                                     "state": "peak_flux"
        #                                   },
        #                                   depends=["state", "mesh_coords"],
        #                                   check_partials=self.check_partials),
        #                    promotes_inputs=[("mesh_coords", "x_em_vol"),
        #                                     ("state", "peak_flux")],
        #                    promotes_outputs=[("max_state:stator", "max_flux_magnitude:stator")])

        core_loss_depends = ["mesh_coords",
                             "temperature",
                             "frequency",
                             #  "max_flux_magnitude:stator",
                             "peak_flux"]

        stator_attrs = self.solvers[0].getOptions(
        )["components"]["stator"]["attrs"]
        rotor_attrs = self.solvers[0].getOptions(
        )["components"]["rotor"]["attrs"]
        print(stator_attrs)
        print(rotor_attrs)
        stator_core_loss_options = {
            "attributes": [*stator_attrs, *rotor_attrs]
        }
        print(stator_core_loss_options)
        self.add_subsystem("stator_core_loss_raw",
                           MachFunctional(solver=self.solvers[0],
                                          func="core_loss",
                                          func_options=stator_core_loss_options,
                                          depends=core_loss_depends,
                                          check_partials=self.check_partials),
                           promotes_inputs=[
                               ("mesh_coords", "x_em_vol"), ("temperature", temperature_name), *core_loss_depends[2:]],
                           promotes_outputs=[("core_loss", "stator_core_loss_raw")])

        self.add_subsystem("stator_core_loss",
                           om.ExecComp(
                               "stator_core_loss = stator_core_loss_raw * stack_length / model_depth"),
                           promotes=["*"])

        # self.add_subsystem("stator_mass_raw",
        #                    MachFunctional(solver=self.solvers[0],
        #                                   func="mass:stator",
        #                                   func_options=stator_core_loss_options,
        #                                   depends=["mesh_coords"],
        #                                   check_partials=self.check_partials),
        #                    promotes_inputs=[("mesh_coords", "x_em_vol")],
        #                    promotes_outputs=[("mass:stator", "stator_mass_raw")])

        # self.add_subsystem("stator_mass",
        #                    om.ExecComp("stator_mass = stator_mass_raw * stack_length / model_depth"),
        #                    promotes=["*"])

        self.add_subsystem("motor_mass_raw",
                           MachFunctional(solver=self.solvers[0],
                                          func="mass:motor",
                                          depends=["mesh_coords",
                                                   "fill_factor"],
                                          check_partials=self.check_partials),
                           promotes_inputs=[
                               ("mesh_coords", "x_em_vol"), "fill_factor"],
                           promotes_outputs=[("mass:motor", "motor_mass_raw")])

        self.add_subsystem("motor_mass",
                           om.ExecComp(
                               "motor_mass = motor_mass_raw * stack_length / model_depth"),
                           promotes=["*"])

        # self.add_subsystem("stator_volume_raw",
        #                    MachFunctional(solver=self.solvers[0],
        #                                   func="volume:stator",
        #                                   func_options=stator_core_loss_options,
        #                                   depends=["mesh_coords"],
        #                                   check_partials=self.check_partials),
        #                    promotes_inputs=[("mesh_coords", "x_em_vol")],
        #                    promotes_outputs=[("volume:stator", "stator_volume_raw")])

        # self.add_subsystem("stator_volume",
        #                    om.ExecComp("stator_volume = stator_volume_raw * stack_length / model_depth"),
        #                    promotes=["*"])

        self.add_subsystem("total_loss",
                           om.ExecComp(
                               "total_loss = ac_loss + dc_loss + stator_core_loss"),
                           promotes=["*"])
        self.add_subsystem("power_out",
                           om.ExecComp(
                               "power_out = average_torque * rpm * pi / 30"),
                           promotes=["*"])
        self.add_subsystem("power_in",
                           om.ExecComp(
                               "power_in = power_out + total_loss"),
                           promotes=["*"])
        self.add_subsystem("efficiency",
                           om.ExecComp(
                               "efficiency = power_out / power_in"),
                           promotes=["*"])

        self.add_subsystem("phase_back_emf",
                           om.ExecComp(
                               "phase_back_emf = 2 * power_out / (3 * (2**0.5)*rms_current)"),
                           promotes=['*'])

        # self.add_subsystem("phase_voltage",
        #                    om.ExecComp(
        #                        "phase_voltage = ((phase_back_emf + stator_phase_resistance * (2**0.5)*rms_current)**2 + (2*pi*L*frequency*(2**0.5)*rms_current)**2)**0.5"),
        #                    promotes=['*'])

        # self.add_subsystem("power_factor",
        #                    om.ExecComp(
        #                        "power_factor = phase_back_emf / phase_voltage"),
        #                    promotes=['*'])

        # TODO: In parallel, adding mach output for permanent magnet demagnetization constraint. Adjust as needed
        """
        pm_demag_depends = ["mesh_coords",
                            "peak_flux"]

        # Set the options
        pm_demag_options = {
            "attributes": self.solvers[0].getOptions()["components"]["magnets"]["attrs"]
        }

        self.add_subsystem("pm_demag",
                           MachFunctional(solver=self.solvers[0],
                                          func="pm_demag",
                                          func_options=pm_demag_options,
                                          depends=pm_demag_depends,
                                          check_partials=self.check_partials),
                           promotes_inputs=[("mesh_coords", "x_em_vol"), *pm_demag_depends[1:]],
                           promotes_outputs=["pm_demag"])
        """

        # # TODO: Change the depends as needed
        # # Mach output for demagnetization proximity using Induced Exponential Aggregation (smooth max) function
        # magnets_attrs = self.solvers[0].getOptions()["components"]["magnets"]["attrs"]
        # self.add_subsystem("demag_proximity",
        #                        MachFunctional(solver=self.solvers[0],
        #                                       func="max_state:magnets",
        #                                       func_options={
        #                                         "rho": 10,
        #                                         "attributes": magnets_attrs,
        #                                         "state": "demag_proximity"
        #                                       },
        #                                       depends=["state", "mesh_coords"], #"flux_density"], including flux density as a depends causes B to be [1,1] exclusively in mach
        #                                       check_partials=self.check_partials),
        #                        promotes_inputs=[("mesh_coords", "x_em_vol"),
        #                                         ("state", "pm_demag_field")],
        #                        promotes_outputs=[("max_state:magnets", "demag_proximity_max:magnets")])

        # uncomment here

        # rotor_core_loss_options = {
        #     "attributes": self.solvers[0].getOptions()["components"]["rotor"]["attrs"]
        # }
        # self.add_subsystem("rotor_core_loss",
        #                    MachFunctional(solver=self.solvers[0],
        #                                   func="core_loss",
        #                                   func_options=rotor_core_loss_options,
        #                                   depends=core_loss_depends),
        #                    promotes_inputs=[("mesh_coords", "x_em_vol"), *core_loss_depends[1:]],
        #                    promotes_outputs=[("core_loss", "rotor_core_loss")])

        # magnet_core_loss_options = {
        #     "attributes": self.solvers[0].getOptions()["components"]["magnets"]["attrs"]
        # }
        # self.add_subsystem("magnet_core_loss",
        #                    MachFunctional(solver=self.solvers[0],
        #                                   func="core_loss",
        #                                   func_options=magnet_core_loss_options,
        #                                   depends=core_loss_depends),
        #                    promotes_inputs=[("mesh_coords", "x_em_vol"), *core_loss_depends[1:]],
        #                    promotes_outputs=[("core_loss", "magnet_core_loss")])


class EMMotorBuilder(Builder):
    def __init__(self,
                 solver_options,
                 warper_type,
                 warper_options,
                 coupled=None,
                 two_dimensional=True,
                 check_partials=False):
        self.solver_options = copy.deepcopy(solver_options)
        self.warper_type = copy.deepcopy(warper_type)
        self.warper_options = copy.deepcopy(warper_options)
        self.coupled = coupled
        self.two_dimensional = two_dimensional
        self.check_partials = check_partials

    def initialize(self, comm):
        # Create PDE solver instance
        self.comm = comm

        npts = len(self.solver_options["multipoint"])
        self.solvers = []
        for i in range(npts):
            solver_options = copy.deepcopy(self.solver_options)
            solver_options.update(self.solver_options["multipoint"][i])
            self.solvers.append(PDESolver(type="magnetostatic",
                                          solver_options=solver_options,
                                          comm=comm))

        if self.two_dimensional:
            self.warper = None
        elif self.warper_type != "idwarp":
            self.warper = MeshWarper(warper_options=self.warper_options,
                                     comm=comm)
        else:
            self.warper = None

        self.state_depends = ["mesh_coords",
                              "temperature",
                              "current_density:phaseA",
                              "current_density:phaseB",
                              "current_density:phaseC"]

    def get_coupling_group_subsystem(self, scenario_name=None):
        return EMMotorCouplingGroup(solvers=self.solvers,
                                    state_depends=self.state_depends,
                                    coupled=self.coupled,
                                    check_partials=self.check_partials,
                                    scenario_name=scenario_name)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return MachMeshGroup(solver=self.solvers[0],
                             warper=self.warper,
                             scenario_name=scenario_name)

    def get_pre_coupling_subsystem(self, scenario_name=None):
        return EMMotorPrecouplingGroup(solvers=self.solvers,
                                       warper=self.warper,
                                       coupled=self.coupled,
                                       scenario_name=scenario_name)

    def get_post_coupling_subsystem(self, scenario_name=None):
        # return None
        return EMMotorOutputsGroup(solvers=self.solvers,
                                   coupled=self.coupled,
                                   check_partials=self.check_partials,
                                   scenario_name=scenario_name)

    def get_number_of_nodes(self):
        """
        Get the number of state nodes on this processor
        """
        num_states = self.get_ndof()
        state_size = self.solvers[0].getStateSize()
        return state_size // num_states

    def get_ndof(self):
        """
        Get the number of states per node
        """
        return self.solvers[0].getNumStates()


if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
    import numpy as np

    # class TestEMState(unittest.TestCase):
    #     from mpi4py import MPI
    #     from omESP import omESP
    #     from motormodel.motors.test.test_motor import _mesh_path, _egads_path, _csm_path, _components, _current_indices, _hallbach_segments
    #     from motormodel.motor_options import _buildSolverOptions

    #     geom = omESP(csm_file=str(_csm_path),
    #                 egads_file=str(_egads_path))
    #     geom_config_values = geom.getConfigurationValues()
    #     num_magnets = int(geom_config_values["num_magnets"])
    #     magnet_divisions = int(geom_config_values["magnet_divisions"])

    #     _, em_options, _ = _buildSolverOptions(_components,
    #                                             [0],
    #                                             magnet_divisions,
    #                                             True,
    #                                             _hallbach_segments,
    #                                             _current_indices)
    #     em_options["mesh"] = {}
    #     em_options["mesh"]["file"] = str(_mesh_path)
    #     em_options["mesh"]["model-file"] = str(_egads_path)

    #     em_options.update(em_options["multipoint"][0])
    #     solver = PDESolver(type="magnetostatic",
    #                        solver_options=em_options,
    #                        comm=MPI.COMM_WORLD)

    #     state_depends = ["mesh_coords",
    #                      "current_density:phaseA",
    #                      "current_density:phaseB",
    #                      "current_density:phaseC"]

    #     # state_depends = ["current_density:phaseA",
    #     #                  "current_density:phaseB",
    #     #                  "current_density:phaseC"]

    #     # state_depends = ["current_density:phaseA"]

    #     def test_em_state_and_flux_mag_group(self):
    #         prob = om.Problem()
    #         prob.model.add_subsystem("state",
    #                                  MachState(solver=self.solver,
    #                                            depends=self.state_depends,
    #                                            check_partials=True),
    #                                  promotes_inputs=[("mesh_coords", "x_em_vol"), *self.state_depends[1:]],
    #                                 #  promotes_inputs=[*self.state_depends],
    #                                  promotes_outputs=[("state", "em_state")])

    #         rotor_attrs = self.solver.getOptions()["components"]["rotor"]["attrs"]
    #         magnet_attrs = self.solver.getOptions()["components"]["magnets"]["attrs"]
    #         airgap_attrs = self.solver.getOptions()["components"]["airgap"]["attrs"]
    #         torque_opts = {
    #             "attributes": [*rotor_attrs, *magnet_attrs],
    #             "axis": [0.0, 0.0, 1.0],
    #             "about": [0.0, 0.0, 0.0],
    #             "air_attributes": airgap_attrs
    #         }

    #         prob.model.add_subsystem("torque",
    #                                  MachFunctional(solver=self.solver,
    #                                                 func="torque",
    #                                                 func_options=torque_opts,
    #                                                 depends=["state", "mesh_coords"],
    #                                                 check_partials=True),
    #                                  promotes_inputs=[("state", "em_state"), ("mesh_coords", "x_em_vol")],
    #                                  promotes_outputs=["torque"])

    #         prob.setup(mode="rev")

    #         # prob.model.state.set_check_partial_options(wrt="*", directional=True)
    #         # prob.model.flux_magnitude.set_check_partial_options(wrt="*", directional=True)

    #         mesh_size = self.solver.getFieldSize("mesh_coords")
    #         mesh_coords = np.zeros(mesh_size)
    #         self.solver.getMeshCoordinates(mesh_coords)

    #         prob["x_em_vol"] = mesh_coords
    #         # prob["current_density:phaseA"] = 0.0
    #         # prob["current_density:phaseB"] = 1e6
    #         # prob["current_density:phaseC"] = -1e6

    #         prob.run_model()

    #         data = prob.check_totals(of=["torque"],
    #                                  wrt=[
    #                                     #   "x_em_vol",
    #                                       "current_density:phaseA",
    #                                       "current_density:phaseB",
    #                                       "current_density:phaseC"])
    #         assert_check_totals(data, atol=1e-6, rtol=1e-6)

    #         # partial_data = prob.check_partials(method="fd", form="central")
    #         # # partial_data = prob.check_partials(method="fd", out_stream=None)
    #         # assert_check_partials(partial_data, atol=np.inf, rtol=1e-5)

    # class TestEMStateAndFluxMagGroup(unittest.TestCase):
    #     from mpi4py import MPI
    #     from omESP import omESP
    #     from motormodel.motors.test.test_motor import _mesh_path, _egads_path, _csm_path, _components, _current_indices, _hallbach_segments
    #     from motormodel.motor_options import _buildSolverOptions

    #     geom = omESP(csm_file=str(_csm_path),
    #                 egads_file=str(_egads_path))
    #     geom_config_values = geom.getConfigurationValues()
    #     num_magnets = int(geom_config_values["num_magnets"])
    #     magnet_divisions = int(geom_config_values["magnet_divisions"])

    #     _, em_options, _ = _buildSolverOptions(_components,
    #                                             [0],
    #                                             magnet_divisions,
    #                                             True,
    #                                             _hallbach_segments,
    #                                             _current_indices)
    #     em_options["mesh"] = {}
    #     em_options["mesh"]["file"] = str(_mesh_path)
    #     em_options["mesh"]["model-file"] = str(_egads_path)

    #     em_options.update(em_options["multipoint"][0])
    #     solver = PDESolver(type="magnetostatic",
    #                        solver_options=em_options,
    #                        comm=MPI.COMM_WORLD)

    #     state_depends = ["mesh_coords",
    #                      "current_density:phaseA",
    #                      "current_density:phaseB",
    #                      "current_density:phaseC"]

    #     def test_em_state_and_flux_mag_group(self):
    #         prob = om.Problem()
    #         prob.model = EMStateAndFluxMagGroup(solver=self.solver,
    #                                             state_depends=self.state_depends,
    #                                             check_partials=True)

    #         prob.setup()

    #         # prob.model.state.set_check_partial_options(wrt=["state", "mesh_coords"], directional=True)
    #         # prob.model.flux_magnitude.set_check_partial_options(wrt="*", directional=True)

    #         mesh_size = self.solver.getFieldSize("mesh_coords")
    #         mesh_coords = np.zeros(mesh_size)
    #         self.solver.getMeshCoordinates(mesh_coords)

    #         prob["x_em_vol"] = mesh_coords
    #         prob["current_density:phaseA"] = 0.0
    #         prob["current_density:phaseB"] = 1
    #         prob["current_density:phaseC"] = -1

    #         prob.run_model()

    #         partial_data = prob.check_partials(method="fd", form="central")
    #         # partial_data = prob.check_partials(method="fd", out_stream=None)
    #         assert_check_partials(partial_data, atol=np.inf, rtol=1e-5)

    class TestACLosses(unittest.TestCase):
        from mpi4py import MPI
        from omESP import omESP
        # from motormodel.motors.test.test_motor import _mesh_path, _egads_path, _csm_path, _components, _current_indices, _hallbach_segments
        from motormodel.motors.pw127e.pw127e import _mesh_path, _egads_path, _csm_path, _components, _current_indices, _hallbach_segments
        from motormodel.motor_options import _buildSolverOptions

        geom = omESP(csm_file=str(_csm_path),
                     egads_file=str(_egads_path))
        geom_config_values = geom.getConfigurationValues()
        num_magnets = int(geom_config_values["num_magnets"])
        magnet_divisions = int(geom_config_values["magnet_divisions"])

        _, em_options, _ = _buildSolverOptions(_components,
                                               [0],
                                               magnet_divisions,
                                               True,
                                               _hallbach_segments,
                                               _current_indices)
        em_options["mesh"] = {}
        em_options["mesh"]["file"] = str(_mesh_path)
        em_options["mesh"]["model-file"] = str(_egads_path)

        em_options.update(em_options["multipoint"][0])
        solver = PDESolver(type="magnetostatic",
                           solver_options=em_options,
                           comm=MPI.COMM_WORLD)

        state_depends = ["mesh_coords",
                         "current_density:phaseA",
                         "current_density:phaseB",
                         "current_density:phaseC"]

        def test_ac_losses(self):
            prob = om.Problem()

            ac_loss_depends = ["mesh_coords",
                               "stack_length",
                               "frequency",
                               "peak_flux",
                               "strand_radius",
                               "model_depth",
                               "strands_in_hand",
                               "num_turns",
                               "num_slots"]
            winding_attrs = self.solver.getOptions(
            )["components"]["windings"]["attrs"]
            prob.model.add_subsystem("ac_loss",
                                     MachFunctional(solver=self.solver,
                                                    func="ac_loss",
                                                    func_options={
                                                        "attributes": winding_attrs},
                                                    depends=ac_loss_depends,
                                                    check_partials=True),
                                     promotes_inputs=[
                                         ("mesh_coords", "x_em_vol"), *ac_loss_depends[1:]],
                                     promotes_outputs=["ac_loss"])

            prob.setup()

            prob.model.ac_loss.set_check_partial_options(
                wrt="*", directional=True)

            mesh_size = self.solver.getFieldSize("mesh_coords")
            mesh_coords = np.zeros(mesh_size)
            self.solver.getMeshCoordinates(mesh_coords)

            peak_flux_size = self.solver.getFieldSize("peak_flux")
            peak_flux = np.zeros(peak_flux_size)
            peak_flux[:] = 2.0

            prob["x_em_vol"] = mesh_coords
            prob["stack_length"] = 0.310
            prob["frequency"] = 1
            prob["peak_flux"] = peak_flux
            prob["strand_radius"] = 0.001671
            prob["strands_in_hand"] = 1
            prob["num_turns"] = 38
            prob["num_slots"] = 27

            prob.run_model()

            # ac_loss_exp = 40638.09914441
            # assert_near_equal(prob["ac_loss"], ac_loss_exp, tolerance=0.005)

            partial_data = prob.check_partials(method="fd", form="central")
            # partial_data = prob.check_partials(method="fd", out_stream=None)
            assert_check_partials(partial_data, atol=1, rtol=1e-5)

            prob.setup()

    unittest.main()
