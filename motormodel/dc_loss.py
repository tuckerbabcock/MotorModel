import openmdao.api as om
import numpy as np

from mach import PDESolver, MachFunctional


class WireLength(om.ExplicitComponent):
    def setup(self):
        self.add_input("num_slots",
                       desc=" The number of slots in the motor",
                       tags=["mphys_input"])
        self.add_input("num_turns",
                       desc=" The number of times the wire has been wrapped around a tooth",
                       tags=["mphys_input"])
        self.add_input("stator_ir",
                       desc=" The inner radius of the stator",
                       tags=["mphys_input"])
        self.add_input("tooth_tip_thickness",
                       desc=" The thickness at the end of the tooth",
                       tags=["mphys_input"])
        self.add_input("slot_depth",
                       desc=" The distance between the the stator inner radius and the edge of the stator yoke",
                       tags=["mphys_input"])
        self.add_input("tooth_width",
                       desc=" The width of the tooth",
                       tags=["mphys_input"])
        self.add_input("stack_length",
                       desc=" The axial depth of the motor",
                       tags=["mphys_input"])

        self.add_output("wire_length",
                        desc=" The length of wire in a single phase")

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        num_slots = inputs["num_slots"][0]
        num_turns = inputs["num_turns"][0]
        stator_ir = inputs["stator_ir"][0]
        tooth_tip_thickness = inputs["tooth_tip_thickness"][0]
        slot_depth = inputs["slot_depth"][0]
        tooth_width = inputs["tooth_width"][0]
        stack_length = inputs["stack_length"][0]

        r_yoke = stator_ir + slot_depth
        r_inner_tooth = stator_ir + tooth_tip_thickness
        slot_width = np.pi * (r_yoke + r_inner_tooth) / num_slots
        # print(f"slot width: {slot_width}")

        # straight sections
        turn_length = 2 * stack_length
        # top/bottom arc sections
        turn_length += 2 * np.pi * ((tooth_width / 2) + (slot_width / 4))

        # print(f"turn length: {turn_length}")
        # total number of turns on all slots
        # length = num_slots * (num_turns / (2*num_slots)) * turn_length
        length = num_slots * num_turns * turn_length
        # print(f"num slots: {num_slots}, num_turns: {num_turns}")

        # plus three 60 deg sections of wire connecting each group of teeth
        r_yoke = stator_ir + slot_depth
        r_inner_tooth = stator_ir + tooth_tip_thickness
        r_avg_winding = (r_yoke + r_inner_tooth) / 2
        length += np.pi * r_avg_winding
        # print(f"wire length: {length}")

        outputs["wire_length"] = length


class DCLoss(om.Group):
    def initialize(self):
        self.options.declare("solver",
                             types=PDESolver,
                             desc="the mach solver object itself",
                             recordable=False)
        self.options.declare("check_partials", default=False)

    def setup(self):
        self.check_partials = self.options["check_partials"]
        self.add_subsystem("wire_length",
                           WireLength(),
                           promotes_inputs=["*"],
                           promotes_outputs=["wire_length"])

        dc_loss_depends = ["mesh_coords",
                           "temperature",
                           "rms_current",
                           "strand_radius",
                           "strands_in_hand",
                           "wire_length"]
        self.add_subsystem("dc_loss",
                           MachFunctional(solver=self.options["solver"],
                                          func="dc_loss",
                                          depends=dc_loss_depends,
                                          check_partials=self.check_partials),
                           promotes_inputs=[
                               ("mesh_coords", "x_em_vol"), *dc_loss_depends[1:]],
                           promotes_outputs=["dc_loss"])


if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

    class TestWireLength(unittest.TestCase):
        def test_wire_length(self):
            prob = om.Problem()
            prob.model = WireLength()
            prob.setup()

            prob["num_slots"] = 27
            prob["num_turns"] = 38
            prob["stator_ir"] = 0.275
            prob["tooth_tip_thickness"] = 0.007
            prob["slot_depth"] = 0.044
            prob["tooth_width"] = 0.030
            prob["stack_length"] = 0.310

            prob.run_model()

            wire_length_exp = 846.46328313
            assert_near_equal(prob["wire_length"],
                              wire_length_exp, tolerance=0.005)

            partial_data = prob.check_partials(method="fd", out_stream=None)
            assert_check_partials(partial_data)

    class TestDCLoss(unittest.TestCase):
        from mpi4py import MPI
        from omESP import omESP
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

        def test_dc_loss(self):
            prob = om.Problem()
            prob.model = DCLoss(solver=self.solver, check_partials=True)

            prob.setup()

            prob.model.dc_loss.set_check_partial_options(
                wrt="*", directional=True)

            mesh_size = self.solver.getFieldSize("mesh_coords")
            mesh_coords = np.zeros(mesh_size)
            self.solver.getMeshCoordinates(mesh_coords)

            prob["num_slots"] = 27
            prob["num_turns"] = 38
            prob["stator_ir"] = 0.275
            prob["tooth_tip_thickness"] = 0.007
            prob["slot_depth"] = 0.044
            prob["tooth_width"] = 0.030
            prob["stack_length"] = 0.310

            prob["x_em_vol"] = mesh_coords
            prob["rms_current"] = 131.58125719
            prob["strand_radius"] = 0.001671
            prob["strands_in_hand"] = 1

            prob.run_model()

            dc_loss_exp = 40638.09914441
            assert_near_equal(prob["dc_loss"], dc_loss_exp, tolerance=0.005)

            partial_data = prob.check_partials(method="fd", form="central")
            # partial_data = prob.check_partials(method="fd", out_stream=None)
            assert_check_partials(partial_data, atol=1, rtol=1e-5)

    unittest.main()
