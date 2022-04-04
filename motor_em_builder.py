import copy
from clingo import SolveControl
import numpy as np
import openmdao.api as om
from mphys import Builder

from mach import PDESolver, MeshWarper
from mach import MachState, MachMeshWarper, MachFunctional, MachMeshGroup

from .maximum_fit import DiscreteInducedExponential

class EMStateAndFluxMagGroup(om.Group):
    def initialize(self):
        self.options.declare("solver", type=PDESolver, recordable=False)
        self.options.declare("depends", types=list)
        self.options.declare("check_partials", default=False)
        self.options.declare("scenario_name", default=None)


    def setup(self):
        self.solver = self.options["solver"]
        self.depends = self.options["depends"]
        self.check_partials = self.options["check_partials"]

        self.add_subsystem("solver",
                           MachState(solver=self.solver,
                                     depends=self.depends,
                                     check_partials=self.check_partials),
                           promotes_inputs=[*self.depends, ("mesh_coords", "x_em_vol")],
                           promotes_outputs=[("state", "em_state")])

        self.add_subsystem("flux_magnitude",
                           MachFunctional(solver=self.solver,
                                          func="flux_magnitude",
                                          check_partials=self.check_partials),
                           promotes_inputs=[("mesh_coords", "x_em_vol")],
                           promotes_outputs=["flux_magnitude"])


class EMMotorCouplingGroup(om.Group):
    def initialize(self):
        self.options.declare("solvers", type=list, recordable=False)
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
            em_states.add_subsystem(f"solver_{idx}",
                                    EMStateAndFluxMagGroup(solver=solver,
                                                           depends=self.depends,
                                                           check_partials=self.check_partials),
                                    promotes_inputs=[*self.depends, ("mesh_coords", "x_em_vol")],
                                    promotes_outputs=[("state", f"em_state_{idx}"),
                                                      ("flux_magnitude", f"flux_magnitude_{idx}")])
        
        self.add_subsystem("peak_flux",
                           DiscreteInducedExponential(num_pts=len(self.solvers)),
                           promotes_outputs=[("data_amplitude", "peak_flux")])

        for idx in range(len(self.solvers)):
            self.connect(f"solver_{idx}.flux_magnitude_{idx}", f"peak_flux.data_{idx}")

        # If coupling to thermal solver, compute heat flux from `peak_flux` ...

class EMMotorPrecouplingGroup:
    """
    Group that handles surface -> volume mesh movement

    To properly support parallel analysis, I'll need to have this component
    partition the input surface coords
    """
    def initialize(self):
        self.options.declare("warper", type=MeshWarper, recordable=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.warper = self.options["warper"]

        # Promote variables with physics-specific tag that MPhys expects
        self.add_subsystem("warper",
                           MachMeshWarper(warper=self.warper),
                           promotes_inputs=[("surf_mesh_coords", "x_em")],
                           promotes_outputs=[("vol_mesh_coords", "x_em_vol")])

class EMMotorOutputsGroup:
    """
    Group that handles calculating outputs after the state solve
    """
    def initialize(self):
        self.options.declare("solvers", type=list, recordable=False)
        self.options.declare("outputs", type=dict, default=None)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solvers = self.options["solvers"]
        self.outputs = self.options["outputs"]
        self.check_partials = self.options["check_partials"]

        # em_functionals = self.add_subsystem("em_functionals", om.ParallelGroup())
        em_functionals = self.add_subsystem("em_functionals", om.Group())
        for idx, solver in enumerate(self.solvers):

            for output in self.outputs:
                if "options" in self.outputs[output]:
                    output_opts = self.outputs[output]["options"]
                else:
                    output_opts = None

                if "depends" in self.outputs[output]:
                    depends = self.outputs[output]["depends"]
                else:
                    depends = []

                em_functionals.add_subsystem(f"{output}_{idx}",
                                             MachFunctional(solver=solver,
                                                            func=output,
                                                            func_options=output_opts,
                                                            depends=depends),
                                             promotes_inputs=[*depends,
                                                              ("mesh_coords", "x_em_vol"),
                                                              ("state", f"em_state_{idx}")],
                                             promotes_outputs=[(output, f"{output}_{idx}")])

class EMMotorBuilder(Builder):
    def __init__(self, solver_options, depends, warper_type, warper_options, outputs, check_partials=False):
        self.solver_options = copy.deepcopy(solver_options)
        self.depends = copy.deepcopy(depends)
        self.warper_type = copy.deepcopy(warper_type)
        self.warper_options = copy.deepcopy(warper_options)
        self.outputs = copy.deepcopy(outputs)
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
        return EMMotorPrecouplingGroup(warper=self.warper,
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
