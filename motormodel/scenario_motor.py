import openmdao.api as om
from mphys.scenario import Scenario
from mphys.coupling_group import CouplingGroup

class ScenarioMotor(Scenario):
    def initialize(self):
        """
        A class to perform a single discipline electromagnetic case.
        The Scenario will add the electromagnetic builder's precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        self.options.declare("em_motor_builder", recordable=False,
                             desc="The Mphys builder for the EM motor solver")
        self.options.declare("thermal_builder", default=None, recordable=False,
                             desc="The Mphys builder for the thermal solver")
        # self.options.declare("in_MultipointParallel", default=False, types=bool,
        #                      desc="Set to `True` if adding this scenario inside a MultipointParallel Group.")
        # self.options.declare("geometry_builder", default=None, recordable=False,
        #                      desc="The optional Mphys builder for the geometry")

    def setup(self):
        em_motor_builder = self.options["em_motor_builder"]
        thermal_builder = self.options["thermal_builder"]
        # geometry_builder = self.options["geometry_builder"]

        # if self.options["in_MultipointParallel"]:
        #     em_motor_builder.initialize(self.comm)

        #     if geometry_builder is not None:
        #         geometry_builder.initialize(self.comm)
        #         self.add_subsystem("mesh", em_motor_builder.get_mesh_coordinate_subsystem(self.name))
        #         self.mphys_add_subsystem("geometry", geometry_builder.get_mesh_coordinate_subsystem(self.name))
        #         self.connect("mesh.x_em0", "geometry.x_em_in")
        #     else:
        #         self.mphys_add_subsystem("mesh",em_motor_builder.get_mesh_coordinate_subsystem(self.name))

        self.mphys_add_pre_coupling_subsystem("em", em_motor_builder, self.name)
        if thermal_builder is not None:
            self.mphys_add_pre_coupling_subsystem("thermal", thermal_builder, self.name)

        coupling_group = CouplingGroup()
        em = em_motor_builder.get_coupling_group_subsystem(self.name)
        coupling_group.mphys_add_subsystem("em", em)

        if thermal_builder is not None:
            thermal = thermal_builder.get_coupling_group_subsystem(self.name)
            coupling_group.mphys_add_subsystem("thermal", thermal)

        coupling_group.nonlinear_solver = om.NonlinearRunOnce()
        coupling_group.linear_solver = om.LinearRunOnce()
        # coupling_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2,
        #                                                       atol=1e-8, rtol=1e-8,
        #                                                       use_aitken=True)
        # coupling_group.linear_solver = om.LinearBlockGS(maxiter=25, iprint=2,
        #                                                 atol=1e-8, rtol=1e-8,
        #                                                 use_aitken=True)

        self.mphys_add_subsystem('coupling', coupling_group)

        self.mphys_add_post_coupling_subsystem("em", em_motor_builder, self.name)
        if thermal_builder is not None:
            self.mphys_add_post_coupling_subsystem("thermal", thermal_builder, self.name)

    def configure(self):
        # connect current densities from pre-coupling to coupling
        em_motor_builder = self.options["em_motor_builder"]
        for idx, _ in enumerate(em_motor_builder.solvers):
            self.connect(f"em_pre.three_phase{idx}.current_density:phaseA",
                         f"solver{idx}.current_density:phaseA")
            self.connect(f"em_pre.three_phase{idx}.current_density:phaseB",
                         f"solver{idx}.current_density:phaseB")
            self.connect(f"em_pre.three_phase{idx}.current_density:phaseC",
                         f"solver{idx}.current_density:phaseC")

        # self.connect("em_pre.slot_area", "slot_area")

        # promote all unconnected inputs from em_pre
        self.promotes("em_pre", any=["num_slots",
                                     "stator_inner_radius",
                                     "tooth_tip_thickness",
                                     "tooth_tip_angle",
                                     "slot_depth",
                                     "slot_radius",
                                     "tooth_width",
                                     "shoe_spacing",
                                     "rms_current_density",
                                     "strand_radius",
                                     "current_density",
                                     "rms_current",
                                     "num_turns",
                                     "strands_in_hand"])

        # promote all unconnected I/O from em_post
        self.promotes("em_post", any=[
                                      "average_torque",
                                      "energy",
                                      "stator_core_loss",
                                      "stator_mass",
                                      "stator_volume",
                                      "num_slots",
                                      "dc_loss",
                                      "efficiency",
                                      "power_in",
                                      "power_out",
                                      "stator_inner_radius",
                                      "tooth_tip_thickness",
                                      "slot_depth",
                                      "tooth_width",
                                      "rpm"
                                     ])

        Scenario.configure(self)
