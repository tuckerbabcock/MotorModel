from mphys.scenario import Scenario

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
        self.mphys_add_subsystem("em_coupling", em_motor_builder.get_coupling_group_subsystem(self.name))
        self.mphys_add_post_coupling_subsystem("em", em_motor_builder, self.name)

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
                                     "rms_current"])

        # promote all unconnected I/O from em_post
        self.promotes("em_post", any=["average_torque",
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
                                      "rpm"])

        Scenario.configure(self)
