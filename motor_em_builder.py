import copy
from clingo import SolveControl
import numpy as np
import openmdao.api as om
from mphys import Builder

from mach import PDESolver, MeshWarper
from mach import MachState, MachMeshWarper, MachFunctional, MachMeshGroup

from average_comp import AverageComp
from maximum_fit import DiscreteInducedExponential
from motor_current import MotorCurrent

class EMStateAndFluxMagGroup(om.Group):
    def initialize(self):
        self.options.declare("solver", types=PDESolver, recordable=False)
        self.options.declare("depends", types=list)
        self.options.declare("check_partials", default=False)
        self.options.declare("scenario_name", default=None)


    def setup(self):
        self.solver = self.options["solver"]
        self.depends = self.options["depends"]
        self.check_partials = self.options["check_partials"]

        self.add_subsystem("state",
                           MachState(solver=self.solver,
                                     depends=self.depends,
                                     check_partials=self.check_partials),
                           promotes_inputs=[*self.depends, ("mesh_coords", "x_em_vol")],
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
        self.options.declare("depends", types=list)
        self.options.declare("check_partials", default=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solvers = self.options["solvers"]
        self.depends = self.options["depends"]
        self.check_partials = self.options["check_partials"]

        # em_states = self.add_subsystem("em_states", om.ParallelGroup())
        em_states = self.add_subsystem("em_states", om.Group())
        for idx, solver in enumerate(self.solvers):
            em_states.add_subsystem(f"solver{idx}",
                                    EMStateAndFluxMagGroup(solver=solver,
                                                           depends=self.depends,
                                                           check_partials=self.check_partials))
                                    # promotes_inputs=[*self.depends])

            self.promotes("em_states",
                        #   inputs=[*[(f"solver{idx}.{input}", f"{input}{idx}") for input in self.depends],
                        #           (f"solver{idx}.x_em_vol", "x_em_vol")],
                          inputs=[*[f"solver{idx}.{input}" for input in self.depends],
                                   (f"solver{idx}.x_em_vol", "x_em_vol")],
                          outputs=[(f"solver{idx}.em_state", f"em_state{idx}")])

        self.add_subsystem("peak_flux",
                           DiscreteInducedExponential(num_pts=len(self.solvers)),
                           promotes_outputs=[("data_amplitude", "peak_flux")])

        for idx, _ in enumerate(self.solvers):
            self.connect(f"em_states.solver{idx}.flux_magnitude", f"peak_flux.data{idx}")
            # self.connect(f"em_states.flux_magnitude{idx}", f"peak_flux.data{idx}")

        # If coupling to thermal solver, compute heat flux from `peak_flux` ...

class EMMotorPrecouplingGroup(om.Group):
    """
    Group that handles surface -> volume mesh movement

    To properly support parallel analysis, I'll need to have this component
    partition the input surface coords
    """
    def initialize(self):
        self.options.declare("solvers", types=list, recordable=False)
        self.options.declare("warper", types=MeshWarper, recordable=False)
        self.options.declare("winding_options", types=dict, desc=" Options for configuring MotorCurrent")
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solvers = self.options["solvers"]
        self.warper = self.options["warper"]

        # Promote variables with physics-specific tag that MPhys expects
        self.add_subsystem("mesh_warper",
                           MachMeshWarper(warper=self.warper),
                           promotes_inputs=[("surf_mesh_coords", "x_em")],
                           promotes_outputs=[("vol_mesh_coords", "x_em_vol")])

        num_slots = self.options["winding_options"]["num_slots"]
        num_turns = self.options["winding_options"]["num_turns"]
        num_strands = self.options["winding_options"]["num_strands"]

        theta_e = []
        for solver in self.solvers:
            solver_options = solver.getOptions()
            theta_e.append(solver_options["theta_e"])

        self.add_subsystem(f"current",
                           MotorCurrent(num_slots=num_slots,
                                        num_turns=num_turns,
                                        num_strands=num_strands,
                                        theta_e=theta_e),
                           promotes_inputs=["*"],
                           promotes_outputs=["slot_area",
                                             "rms_current",
                                             "three_phase*.current_density:phase*"])

class EMMotorOutputsGroup(om.Group):
    """
    Group that handles calculating outputs after the state solve
    """
    def initialize(self):
        self.options.declare("solvers", types=list, recordable=False)
        self.options.declare("outputs", types=dict, default=None)
        self.options.declare("check_partials", default=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solvers = self.options["solvers"]
        self.outputs = self.options["outputs"]
        self.check_partials = self.options["check_partials"]

        # em_functionals = self.add_subsystem("em_functionals", om.ParallelGroup())
        torque = self.add_subsystem("torque", om.Group())
        for idx, solver in enumerate(self.solvers):

            rotor_attrs = solver.getOptions()["components"]["rotor"]["attr"]
            magnet_attrs = solver.getOptions()["components"]["magnets"]["attrs"]
            torque_opts = {
                "attributes": [rotor_attrs, *magnet_attrs],
                "axis": [0.0, 0.0, 1.0],
                "about": [0.0, 0.0, 0.0]
            }

            torque.add_subsystem(f"torque{idx}",
                                 MachFunctional(solver=solver,
                                                func="torque",
                                                func_options=torque_opts,
                                                depends=["state", "mesh_coords"]))

            self.promotes("torque",
                          inputs=[(f"torque{idx}.mesh_coords", "x_em_vol"),
                                  (f"torque{idx}.state", f"em_state{idx}")])

        self.add_subsystem("avg_torque",
                           AverageComp(num_pts=len(self.solvers)),
                           promotes_outputs=[("data_average", "average_torque")])

        for idx, _ in enumerate(self.solvers):
            self.connect(f"torque.torque{idx}.torque", f"avg_torque.data{idx}")

        ac_depends = ["mesh_coords", "stack_length", "slot_area", "frequency", "peak_flux", "strand_radius"]
        self.add_subsystem("ac_loss",
                           MachFunctional(solver=solver,
                                          func="ac_loss",
                                          depends=ac_depends),
                           promotes_inputs=[("mesh_coords", "x_em_vol"), *ac_depends[1:]],
                           promotes_outputs=["ac_loss"])

class EMMotorBuilder(Builder):
    def __init__(self,
                 solver_options,
                 depends,
                 warper_type,
                 warper_options,
                 outputs,
                 winding_options,
                 check_partials=False):
        self.solver_options = copy.deepcopy(solver_options)
        self.depends = copy.deepcopy(depends)
        self.warper_type = copy.deepcopy(warper_type)
        self.warper_options = copy.deepcopy(warper_options)
        self.outputs = copy.deepcopy(outputs)
        self.winding_options = copy.deepcopy(winding_options)
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

        if (self.warper_type != "idwarp"):
            self.warper = MeshWarper(warper_options=self.warper_options,
                                     comm=comm)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return EMMotorCouplingGroup(solvers=self.solvers,
                                    depends=self.depends,
                                    check_partials=self.check_partials,
                                    scenario_name=scenario_name)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return MachMeshGroup(solver=self.solvers[0],
                             warper=self.warper,
                             scenario_name=scenario_name)

    def get_pre_coupling_subsystem(self, scenario_name=None):
        return EMMotorPrecouplingGroup(solvers=self.solvers,
                                       warper=self.warper,
                                       winding_options=self.winding_options,
                                       scenario_name=scenario_name)

    def get_post_coupling_subsystem(self, scenario_name=None):
        return EMMotorOutputsGroup(solvers=self.solvers,
                                   outputs=self.outputs,
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
