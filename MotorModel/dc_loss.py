import openmdao.api as om
import numpy as np

from mach import PDESolver, MachFunctional
from setuptools import setup

# dc_loss.add_subsystem("slot_width",
#                         om.ExecComp("slot_width = np.pi * (2*stator_inner_radius + slot_depth + tooth_tip_thickness) / num_slots"),
#                         promotes=["*"])

# dc_loss.add_subsystem("turn_length",
#                         om.ExecComp("turn_length = (2 * stack_length + np.pi*((tooth_width / 2) + (slot_width / 4)))"))
# dc_loss.add_subsystem("wire_length",
#                         om.ExecComp("wire_length = num_turns * turn_length + "))
#                             # MachFunctional(solver=self.solvers[0],
#                             #                 func="dc_loss",
#                             #                 depends=dc_loss_depends),
#                             # promotes_inputs=[("mesh_coords", "x_em_vol"), *dc_loss_depends[1:]],
#                             # promotes_outputs=["dc_loss"])

class WireLength(om.ExplicitComponent):
    def setup(self):
        self.add_input("num_slots",
                       desc=" The number of slots in the motor")
        self.add_input("num_turns",
                       desc=" The number of times the wire has been wrapped around a tooth")
        self.add_input("stator_inner_radius",
                       desc=" The inner radius of the stator")
        self.add_input("tooth_tip_thickness",
                       desc=" The thickness at the end of the tooth")
        self.add_input("slot_depth",
                       desc=" The distance between the the stator inner radius and the edge of the stator yoke")
        self.add_input("tooth_width",
                       desc=" The width of the tooth")
        self.add_input("stack_length",
                       desc=" The axial depth of the motor")

        self.add_output("wire_length",
                        desc=" The length of wire in a single phase")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        num_slots = inputs["num_slots"][0]
        num_turns = inputs["num_turns"][0]
        stator_inner_radius = inputs["stator_inner_radius"][0]
        tooth_tip_thickness = inputs["tooth_tip_thickness"][0]
        slot_depth = inputs["slot_depth"][0]
        tooth_width = inputs["tooth_width"][0]
        stack_length = inputs["stack_length"][0]

        r_yoke = stator_inner_radius + slot_depth;
        r_inner_tooth = stator_inner_radius + tooth_tip_thickness;
        slot_width = np.pi * (r_yoke + r_inner_tooth) / num_slots;
        print(f"slot width: {slot_width}")

        # straight sections
        turn_length = stack_length;
        # top/bottom arc sections
        turn_length += np.pi * ((tooth_width / 2) + (slot_width / 4));

        print(f"turn length: {turn_length}")
        # total number of turns on all slots
        # length = num_slots * (num_turns / (2*num_slots)) * turn_length 
        length = num_turns * turn_length 
        # print(f"num slots: {num_slots}, num_turns: {num_turns}")

        # plus three 60 deg sections of wire connecting each group of teeth
        r_yoke = stator_inner_radius + slot_depth;
        r_inner_tooth = stator_inner_radius + tooth_tip_thickness;
        r_avg_winding = (r_yoke + r_inner_tooth) / 2;
        length += np.pi * r_avg_winding;
        print(f"wire length: {length}")

        outputs["wire_length"] = length

class DCLoss(om.Group):
    def initialize(self):
        self.options.declare("solver",
                             types=PDESolver,
                             desc="the mach solver object itself",
                             recordable=False)

    def setup(self):
        self.add_subsystem("wire_length",
                           WireLength(),
                           promotes_inputs=["*"],
                           promotes_outputs=["wire_length"])

        dc_loss_depends = ["mesh_coords",
                           "rms_current",
                           "strand_radius",
                           "strands_in_hand",
                           "wire_length"]
        self.add_subsystem("dc_loss",
                            MachFunctional(solver=self.options["solver"],
                                           func="dc_loss",
                                           depends=dc_loss_depends),
                            promotes_inputs=[("mesh_coords", "x_em_vol"), *dc_loss_depends[1:]],
                            promotes_outputs=["dc_loss"])