import copy

import openmdao.api as om
from mphys import Builder

from mach import PDESolver, MeshWarper
from mach import MachState, MachMeshWarper, MachFunctional, MachMeshGroup

from .average_comp import AverageComp
from .maximum_fit import DiscreteInducedExponential
from .motor_current import MotorCurrent
from .dc_loss import WireLength, DCLoss

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
                           promotes_inputs=[("mesh_coords", "x_em_vol"), *depends[1:]],
                           promotes_outputs=[("state", "em_state")])

        self.add_subsystem("flux_magnitude",
                           MachFunctional(solver=self.solver,
                                          func="flux_magnitude",
                                          check_partials=self.check_partials,
                                          depends=["state", "mesh_coords"]),
                           promotes_inputs=[("state", "em_state"), ("mesh_coords", "x_em_vol")],
                           promotes_outputs=["flux_magnitude"])


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

        # em_states = self.add_subsystem("em_states", om.ParallelGroup())
        em_states = self.add_subsystem("em_states", om.Group())
        for idx, solver in enumerate(self.solvers):
            em_states.add_subsystem(f"solver{idx}",
                                    EMStateAndFluxMagGroup(solver=solver,
                                                           state_depends=depends,
                                                           check_partials=self.check_partials))

            self.promotes("em_states",
                          inputs=[*[f"solver{idx}.{input}" for input in depends[1:]],
                                   (f"solver{idx}.x_em_vol", "x_em_vol")],
                          outputs=[(f"solver{idx}.em_state", f"em_state{idx}")])

        self.add_subsystem("peak_flux",
                           DiscreteInducedExponential(num_pts=len(self.solvers),
                                                      rho=100),
                           promotes_outputs=[("data_amplitude", "peak_flux")])

        for idx, _ in enumerate(self.solvers):
            self.connect(f"em_states.solver{idx}.flux_magnitude", f"peak_flux.data{idx}")

        coupled = self.options["coupled"]
        # If coupling to thermal solver, compute heat sources...
        if coupled == "thermal":

            self.add_subsystem("stator_max_flux_magnitude",
                           MachFunctional(solver=self.solvers[0],
                                          func="max_flux_magnitude:stator",
                                          func_options={"rho": 50, "attributes": [1]},
                                          depends=["state", "mesh_coords"]),
                           promotes_inputs=[("mesh_coords", "x_em_vol"),
                                            ("state", "em_state0")],
                           promotes_outputs=["max_flux_magnitude:stator"])

            self.add_subsystem("wire_length",
                               WireLength(),
                               promotes_inputs=["*"],
                               promotes_outputs=["wire_length"])

            heat_source_inputs = [("mesh_coords", "x_em_vol"),
                                  "frequency",
                                  "max_flux_magnitude:stator",
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
                                              func_options={"space-dis": {
                                                                "basis-type": "h1",
                                                                "degree": 1
                                                            }},
                                              depends=heat_source_inputs),
                               promotes_inputs=heat_source_inputs,
                               promotes_outputs=[("heat_source", "thermal_load")])

        elif coupled is not None:
            raise ValueError("EM Motor builder only supports coupling with a thermal solver")

class EMMotorPrecouplingGroup(om.Group):
    """
    Group that handles surface -> volume mesh movement

    To properly support parallel analysis, I'll need to have this component
    partition the input surface coords
    """
    def initialize(self):
        self.options.declare("solvers", types=list, recordable=False)
        self.options.declare("warper", recordable=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solvers = self.options["solvers"]
        self.warper = self.options["warper"]

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
                                             "three_phase*.current_density:phase*"])

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

        # em_functionals = self.add_subsystem("em_functionals", om.ParallelGroup())
        torque = self.add_subsystem("torque", om.Group())
        for idx, solver in enumerate(self.solvers):

            rotor_attrs = solver.getOptions()["components"]["rotor"]["attrs"]
            magnet_attrs = solver.getOptions()["components"]["magnets"]["attrs"]
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
                                                depends=["state", "mesh_coords"]))

            self.promotes("torque",
                          inputs=[(f"torque{idx}.mesh_coords", "x_em_vol"),
                                  (f"torque{idx}.state", f"em_state{idx}")])

        self.add_subsystem("raw_avg_torque",
                           AverageComp(num_pts=len(self.solvers)),
                           promotes_outputs=[("data_average", "raw_average_torque")])

        for idx, _ in enumerate(self.solvers):
            self.connect(f"torque.torque{idx}.torque", f"raw_avg_torque.data{idx}")

        self.add_subsystem("avg_torque",
                           om.ExecComp("average_torque = raw_average_torque * stack_length / model_depth"),
                           promotes=["*"])

        airgap_attrs = solver.getOptions()["components"]["airgap"]["attrs"]
        self.add_subsystem("raw_energy",
                           MachFunctional(solver=self.solvers[0],
                                          func="energy",
                                          func_options={"attributes": airgap_attrs},
                                          depends=["state", "mesh_coords"]),
                           promotes_inputs=[("mesh_coords", "x_em_vol"),
                                            ("state", "em_state0")],
                           promotes_outputs=[("energy", "raw_energy")])

        self.add_subsystem("mag_energy",
                           om.ExecComp("energy = raw_energy * stack_length / model_depth"),
                           promotes=["*"])


        coupled = self.options["coupled"]
        # If not coupling to thermal solver, compute stator max flux here post coupling
        if coupled != "thermal":
            stator_attrs = self.solvers[0].getOptions()["components"]["stator"]["attrs"]
            self.add_subsystem("stator_max_flux_magnitude",
                                MachFunctional(solver=self.solvers[0],
                                               func="max_flux_magnitude:stator",
                                               func_options={"rho": 50, "attributes": stator_attrs},
                                               depends=["state", "mesh_coords"]),
                                promotes_inputs=[("mesh_coords", "x_em_vol"),
                                                 ("state", "em_state0")],
                                promotes_outputs=["max_flux_magnitude:stator"])

        # # winding_attrs = self.solvers[0].getOptions()["components"]["windings"]["attrs"]
        # # self.add_subsystem("winding_max_flux_magnitude",
        # #                    MachFunctional(solver=self.solvers[0],
        # #                                   func="max_flux_magnitude:winding",
        # #                                   func_options={"rho": 50, "attributes": winding_attrs},
        # #                                   depends=["state", "mesh_coords"]),
        # #                    promotes_inputs=[("mesh_coords", "x_em_vol"),
        # #                                     ("state", "em_state0")],
        # #                    promotes_outputs=["max_flux_magnitude:winding"])

        airgap_attrs = self.solvers[0].getOptions()["components"]["airgap"]["attrs"]
        self.add_subsystem("airgap_average_flux_magnitude",
                           MachFunctional(solver=self.solvers[0],
                                          func="average_flux_magnitude:airgap",
                                          func_options={"attributes": airgap_attrs},
                                          depends=["state", "mesh_coords"]),
                           promotes_inputs=[("mesh_coords", "x_em_vol"),
                                            ("state", "em_state0")],
                           promotes_outputs=["average_flux_magnitude:airgap"])

        ac_loss_depends = ["mesh_coords",
                           "stack_length",
                           "frequency",
                           "peak_flux",
                           "strand_radius",
                           "model_depth",
                           "strands_in_hand",
                           "num_turns",
                           "num_slots"]

        winding_attrs = self.solvers[0].getOptions()["components"]["windings"]["attrs"]
        self.add_subsystem("ac_loss",
                           MachFunctional(solver=self.solvers[0],
                                          func="ac_loss",
                                          func_options={"attributes": winding_attrs},
                                          depends=ac_loss_depends),
                           promotes_inputs=[("mesh_coords", "x_em_vol"), *ac_loss_depends[1:]],
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
                                            "strands_in_hand"],
                           promotes_outputs=["*"])


        self.add_subsystem("winding_max_peak_flux",
                           MachFunctional(solver=self.solvers[0],
                                          func="max_state",
                                          func_options={
                                              "rho": 50,
                                              "attributes": winding_attrs,
                                              "state": "flux_magnitude"
                                          },
                                          depends=["mesh_coords", "state"]),
                           promotes_inputs=[("mesh_coords", "x_em_vol"),
                                            ("state", "peak_flux")],
                           promotes_outputs=[("max_state", "winding_max_peak_flux")])

        core_loss_depends = ["mesh_coords",
                             "frequency",
                             "max_flux_magnitude:stator"]

        stator_core_loss_options = {
            "attributes": self.solvers[0].getOptions()["components"]["stator"]["attrs"]
        }
        self.add_subsystem("stator_core_loss_raw",
                           MachFunctional(solver=self.solvers[0],
                                          func="core_loss",
                                          func_options=stator_core_loss_options,
                                          depends=core_loss_depends),
                           promotes_inputs=[("mesh_coords", "x_em_vol"), *core_loss_depends[1:]],
                           promotes_outputs=[("core_loss", "stator_core_loss_raw")])

        self.add_subsystem("stator_core_loss",
                           om.ExecComp("stator_core_loss = stator_core_loss_raw * stack_length / model_depth"),
                           promotes=["*"])

        self.add_subsystem("stator_mass_raw",
                           MachFunctional(solver=self.solvers[0],
                                          func="mass:stator",
                                          func_options=stator_core_loss_options,
                                          depends=["mesh_coords"]),
                           promotes_inputs=[("mesh_coords", "x_em_vol")],
                           promotes_outputs=[("mass:stator", "stator_mass_raw")])

        self.add_subsystem("stator_mass",
                           om.ExecComp("stator_mass = stator_mass_raw * stack_length / model_depth"),
                           promotes=["*"])

        self.add_subsystem("stator_volume_raw",
                           MachFunctional(solver=self.solvers[0],
                                          func="volume:stator",
                                          func_options=stator_core_loss_options,
                                          depends=["mesh_coords"]),
                           promotes_inputs=[("mesh_coords", "x_em_vol")],
                           promotes_outputs=[("volume:stator", "stator_volume_raw")])

        self.add_subsystem("stator_volume",
                           om.ExecComp("stator_volume = stator_volume_raw * stack_length / model_depth"),
                           promotes=["*"])

        self.add_subsystem("total_loss",
                           om.ExecComp("total_loss = ac_loss + dc_loss + stator_core_loss"),
                           promotes=["*"])
        self.add_subsystem("power_out",
                           om.ExecComp("power_out = average_torque * rpm * pi / 30"),
                           promotes=["*"])
        self.add_subsystem("power_in",
                           om.ExecComp("power_in = power_out + total_loss"),
                           promotes=["*"])
        self.add_subsystem("efficiency",
                           om.ExecComp("efficiency = power_out / power_in"),
                           promotes=["*"])
        ###### uncomment here

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
