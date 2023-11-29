import numpy as np

import openmdao.api as om


def _greens_theorem_integral_segment(p1, p2):
    return 0.5*(p2[1]*p1[0] - p2[0]*p1[1])


def _greens_theorem_integral_arc(c, r, t0, t1):
    return 0.5 * \
        (c[0]*r*(np.sin(t1) - np.sin(t0))
         - c[1]*r*(np.cos(t1) - np.cos(t0))
         + r**2*(t1 - t0))


class StatorYokeFilletAngle(om.ImplicitComponent):
    def setup(self):
        self.add_input("stator_or",
                       desc=" The outer radius of the stator")
        self.add_input("stator_ir",
                       desc=" The inner radius of the stator")
        self.add_input("slot_depth",
                       desc=" The distance between the the stator inner radius and the edge of the stator yoke")
        self.add_input("tooth_width",
                       desc=" The width of the tooth")
        self.add_input("tooth_tip_thickness",
                       desc=" The thickness at the end of the tooth")
        self.add_input("slot_radius_stator",
                       desc=" The radius of the fillet between the tooth and stator yoke")

        self.add_output("theta", val=-np.pi/2,
                        desc=" The angle of intersection between the inside of the stator yoke and the stator yoke fillet")

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals):
        stator_or = inputs["stator_or"]
        stator_ir = inputs["stator_ir"]
        slot_depth = inputs["slot_depth"]
        tooth_width = inputs["tooth_width"]
        tooth_tip_thickness = inputs["tooth_tip_thickness"]
        slot_radius_stator = inputs["slot_radius_stator"]

        stator_yoke_thickness = stator_or - \
            (stator_ir + tooth_tip_thickness + slot_depth)
        srs_center_x = tooth_width / 2 + slot_radius_stator

        theta = outputs["theta"]
        residuals["theta"] = srs_center_x + slot_radius_stator * \
            np.cos(theta) - (stator_or - stator_yoke_thickness)*np.cos(theta)


class ToothLengths(om.ExplicitComponent):
    def setup(self):
        self.add_input("stator_or",
                       desc=" The outer radius of the stator")
        self.add_input("stator_ir",
                       desc=" The inner radius of the stator")
        self.add_input("slot_depth",
                       desc=" The distance between the the stator inner radius and the edge of the stator yoke")
        self.add_input("tooth_width",
                       desc=" The width of the tooth")
        self.add_input("tooth_tip_thickness",
                       desc=" The thickness at the end of the tooth")
        self.add_input("tooth_tip_angle",
                       desc=" The angle between the flat on the back of the shoe and the horizontal")
        self.add_input("slot_radius_stator",
                       desc=" The radius of the fillet between the tooth and stator yoke")
        self.add_input("slot_radius_tooth",
                       desc=" The radius of the fillet between the tooth and tooth tip")
        self.add_input("shoe_spacing",
                       desc=" The spacing (length) between teeth tips")
        self.add_input("num_slots",
                       desc=" The number of slots in the stator")
        self.add_input("coolant_thickness", val=0.0,
                       desc=" The thickness of the in-slot-coolant (if there)")
        self.add_input("stator_yoke_fillet_angle",
                       desc=" The angle of intersection between the inside of the stator yoke and the stator yoke fillet")

        self.add_output("shoe_inner_length",
                        desc=" The length of the angled segment on the inner shoe")
        self.add_output("tooth_length",
                        desc=" The length of the vertical segment on the tooth")
        self.add_output("stator_yoke_arc_length",
                        desc=" The arc length along the inside of the stator yoke")
        self.add_output("slot_area",
                        desc=" The winding area in the slot")

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        stator_or = inputs["stator_or"]
        stator_ir = inputs["stator_ir"]
        slot_depth = inputs["slot_depth"]
        tooth_width = inputs["tooth_width"]
        tooth_tip_thickness = inputs["tooth_tip_thickness"]
        tooth_tip_angle = inputs["tooth_tip_angle"]
        slot_radius_stator = inputs["slot_radius_stator"]
        slot_radius_tooth = inputs["slot_radius_tooth"]
        shoe_spacing = inputs["shoe_spacing"]
        num_slots = inputs["num_slots"]
        coolant_thickness = inputs["coolant_thickness"]
        theta = inputs["stator_yoke_fillet_angle"]

        stator_yoke_thickness = stator_or - \
            (stator_ir + tooth_tip_thickness + slot_depth)
        shoe_spacing_angle = shoe_spacing * 360 / (2 * np.pi*stator_ir)
        shoe_angle = 360 / num_slots - shoe_spacing_angle

        pt1_r = stator_ir + tooth_tip_thickness
        pt1_theta = (-90 + shoe_angle/2)*np.pi / 180

        pt1_x = pt1_r * np.cos(pt1_theta)
        pt1_y = pt1_r * np.sin(pt1_theta)

        srt_center_x = tooth_width / 2 + slot_radius_tooth
        theta_1 = (shoe_angle / 2 + tooth_tip_angle) * np.pi / 180

        pt2_x = srt_center_x + slot_radius_tooth * np.cos(np.pi/2 + theta_1)
        pt2_y = pt1_y - np.tan(theta_1) * (pt1_x - pt2_x)

        # print(f"pt1: [{pt1_x}, {pt1_y}], pt2: [{pt2_x}, {pt2_y}]")

        outputs["shoe_inner_length"] = (pt1_x - pt2_x) / np.cos(theta_1)
        # outputs["shoe_inner_length"] = np.sqrt(
        #     ((pt1_x - pt2_x)**2 + (pt1_y - pt2_y)**2))

        srt_center_y = pt2_y - slot_radius_tooth * np.sin(np.pi/2 + theta_1)

        pt3_x = tooth_width / 2
        pt3_y = srt_center_y

        srs_center_y = (stator_or - stator_yoke_thickness) * \
            np.sin(theta) - slot_radius_stator*np.sin(theta)
        pt4_x = tooth_width / 2
        pt4_y = srs_center_y
        outputs["tooth_length"] = pt3_y - pt4_y

        # print(f"pt3: [{pt3_x}, {pt3_y}], pt4: [{pt4_x}, {pt4_y}]")

        outputs["stator_yoke_arc_length"] = (stator_or - stator_yoke_thickness) * \
            (-np.pi/2 + np.pi/num_slots - theta) - coolant_thickness/2

        outputs["slot_area"] = _greens_theorem_integral_segment(np.array([pt1_x, pt1_y]),
                                                                np.array([pt2_x, pt2_y]))

        outputs["slot_area"] += _greens_theorem_integral_arc(np.array([srt_center_x, srt_center_y]),
                                                             slot_radius_tooth,
                                                             np.pi/2 + theta_1,
                                                             np.pi)

        outputs["slot_area"] += _greens_theorem_integral_segment(np.array([pt3_x, pt3_y]),
                                                                 np.array([pt4_x, pt4_y]))

        outputs["slot_area"] += _greens_theorem_integral_arc(np.array([srt_center_x, srs_center_y]),
                                                             slot_radius_stator,
                                                             -np.pi,
                                                             theta)

        coolant_thickness_angle = (-np.pi/2 + np.pi/num_slots -
                                   coolant_thickness/2/(stator_or - stator_yoke_thickness))
        outputs["slot_area"] += _greens_theorem_integral_arc(np.array([0.0, 0.0]),
                                                             (stator_or -
                                                                 stator_yoke_thickness),
                                                             theta,
                                                             coolant_thickness_angle)

        pt6 = (stator_or - stator_yoke_thickness) * \
            np.array([np.cos(coolant_thickness_angle),
                     np.sin(coolant_thickness_angle)])
        pt7 = (stator_ir + tooth_tip_thickness) * \
            np.array([np.cos(coolant_thickness_angle),
                     np.sin(coolant_thickness_angle)])
        outputs["slot_area"] += _greens_theorem_integral_segment(pt6, pt7)

        outputs["slot_area"] += _greens_theorem_integral_arc(np.array([0.0, 0.0]),
                                                             (stator_ir +
                                                                 tooth_tip_thickness),
                                                             coolant_thickness_angle,
                                                             pt1_theta)


class ValidLengths(om.Group):
    def setup(self):
        self.add_subsystem("intersect_angle",
                           StatorYokeFilletAngle(),
                           promotes_inputs=["stator_or",
                                            "stator_ir",
                                            "slot_depth",
                                            "tooth_width",
                                            "tooth_tip_thickness",
                                            "slot_radius_stator"])

        self.add_subsystem("tooth_lengths",
                           ToothLengths(),
                           promotes_inputs=["stator_or",
                                            "stator_ir",
                                            "slot_depth",
                                            "tooth_width",
                                            "tooth_tip_thickness",
                                            "tooth_tip_angle",
                                            "slot_radius_stator",
                                            "slot_radius_tooth",
                                            "shoe_spacing",
                                            "num_slots",
                                            "coolant_thickness"],
                           promotes_outputs=["shoe_inner_length",
                                             "tooth_length",
                                             "stator_yoke_arc_length",
                                             "slot_area"])

        self.connect("intersect_angle.theta",
                     "tooth_lengths.stator_yoke_fillet_angle")

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 20
        self.linear_solver = om.DirectSolver()


if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_check_partials

    class TestGreensTheoremArea(unittest.TestCase):
        def test_segment_rectangle(self):

            length = 10
            width = 0.5
            p1 = np.array([0.0, 0.0])
            p2 = np.array([length, 0.0])
            p3 = np.array([length, width])
            p4 = np.array([0.0, width])

            theta = np.pi/4
            R = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
            p1 = R @ p1
            p2 = R @ p2
            p3 = R @ p3
            p4 = R @ p4

            area = _greens_theorem_integral_segment(p1, p2)
            area += _greens_theorem_integral_segment(p2, p3)
            area += _greens_theorem_integral_segment(p3, p4)
            area += _greens_theorem_integral_segment(p4, p1)

            self.assertAlmostEqual(length*width, area)

        def test_circle(self):

            radius = 2
            c = np.array([1.0, -1.0])

            area = _greens_theorem_integral_arc(c, radius, 0, 2*np.pi)

            self.assertAlmostEqual(np.pi * radius**2, area)

        def test_rounded_box(self):

            radius = 1
            p1 = np.array([0.0, 0.0])
            p2 = np.array([2*radius, 0.0])
            p3 = np.array([2*radius, 2*radius])
            p4 = np.array([0.0, 2*radius])

            c = np.array([2*radius, radius])

            area = _greens_theorem_integral_segment(p1, p2)
            area += _greens_theorem_integral_arc(c, radius, -np.pi/2, np.pi/2)
            area += _greens_theorem_integral_segment(p3, p4)
            area += _greens_theorem_integral_segment(p4, p1)

            self.assertAlmostEqual((2*radius)**2 + 0.5*np.pi*radius**2, area)

        def test_circle_box(self):

            radius = 1
            c1 = np.array([radius, 0.0])
            c2 = np.array([0.0, radius])
            c3 = np.array([-radius, 0.0])
            c4 = np.array([0.0, -radius])

            area = _greens_theorem_integral_arc(c1, radius, -np.pi/2, np.pi/2)
            area += _greens_theorem_integral_arc(c2, radius, 0, np.pi)
            area += _greens_theorem_integral_arc(c3, radius, np.pi/2, -np.pi/2)
            area += _greens_theorem_integral_arc(c4, radius, -np.pi, 2*np.pi)

            self.assertAlmostEqual((2*radius)**2 + 2*np.pi*radius**2, area)

    class TestStatorYokeFilletAngle(unittest.TestCase):
        def test_stator_yoke_fillet_angle(self):

            problem = om.Problem()
            problem.model.add_subsystem("intersect_angle",
                                        StatorYokeFilletAngle(),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["theta"])

            problem.model.nonlinear_solver = om.NewtonSolver(
                solve_subsystems=False)
            problem.model.nonlinear_solver.options['iprint'] = 2
            problem.model.nonlinear_solver.options['maxiter'] = 20
            problem.model.linear_solver = om.DirectSolver()

            problem.setup()

            problem["stator_or"] = 0.659/2
            problem["stator_ir"] = 0.550/2
            problem["slot_depth"] = 0.044
            problem["tooth_width"] = 0.0125
            problem["tooth_tip_thickness"] = 0.007
            problem["slot_radius_stator"] = 0.005

            problem.run_model()

            problem.model.list_inputs(units=True, prom_name=True)
            problem.model.list_outputs(
                residuals=True, units=True, prom_name=True)

            self.assertAlmostEqual(-1.5357424231, problem["theta"][0])

        def test_stator_yoke_fillet_angle_partials(self):

            problem = om.Problem()
            problem.model.add_subsystem("intersect_angle",
                                        StatorYokeFilletAngle(),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["theta"])

            problem.setup()

            problem["stator_or"] = 0.659/2
            problem["stator_ir"] = 0.550/2
            problem["slot_depth"] = 0.044
            problem["tooth_width"] = 0.0125
            problem["tooth_tip_thickness"] = 0.007
            problem["slot_radius_stator"] = 0.005

            problem.run_model()

            data = problem.check_partials(form="central")
            assert_check_partials(data)

    class TestToothLengths(unittest.TestCase):
        def test_tooth_lengths(self):

            problem = om.Problem()
            problem.model.add_subsystem("tooth_lengths",
                                        ToothLengths(),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["*"])

            problem.setup()

            problem["stator_or"] = 0.659/2
            problem["stator_ir"] = 0.550/2
            problem["slot_depth"] = 0.044
            problem["tooth_width"] = 0.0125
            problem["tooth_tip_thickness"] = 0.007
            problem["tooth_tip_angle"] = 10
            problem["slot_radius_stator"] = 0.005
            problem["slot_radius_tooth"] = 0.005
            problem["shoe_spacing"] = 0.01
            problem["num_slots"] = 27
            problem["coolant_thickness"] = 0.01
            problem["stator_yoke_fillet_angle"] = -1.5357424231

            problem.run_model()

            problem.model.list_inputs(units=True, prom_name=True)
            problem.model.list_outputs(
                residuals=True, units=True, prom_name=True)

            self.assertAlmostEqual(0.018417781400324195,
                                   problem["shoe_inner_length"][0])
            self.assertAlmostEqual(0.03038482205330717,
                                   problem["tooth_length"][0])
            self.assertAlmostEqual(0.02150424858700651,
                                   problem["stator_yoke_arc_length"][0])
            self.assertAlmostEqual(0.00102041,
                                   problem["slot_area"][0])

        def test_tooth_lengths_partials(self):

            problem = om.Problem()
            problem.model.add_subsystem("tooth_lengths",
                                        ToothLengths(),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["*"])

            problem.setup()

            problem["stator_or"] = 0.659/2
            problem["stator_ir"] = 0.550/2
            problem["slot_depth"] = 0.044
            problem["tooth_width"] = 0.0125
            problem["tooth_tip_thickness"] = 0.007
            problem["tooth_tip_angle"] = 10
            problem["slot_radius_stator"] = 0.005
            problem["slot_radius_tooth"] = 0.005
            problem["shoe_spacing"] = 0.01
            problem["coolant_thickness"] = 0.01
            problem["num_slots"] = 27
            problem["stator_yoke_fillet_angle"] = -1.5357424231

            problem.run_model()

            data = problem.check_partials(form="central")
            assert_check_partials(data)

    class TestValidLengths(unittest.TestCase):
        def test_valid_lengths(self):

            problem = om.Problem()
            problem.model.add_subsystem("valid_lengths",
                                        ValidLengths(),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["*"])

            problem.setup()

            problem["stator_or"] = 0.659/2
            problem["stator_ir"] = 0.550/2
            problem["slot_depth"] = 0.044
            problem["tooth_width"] = 0.0125
            problem["tooth_tip_thickness"] = 0.007
            problem["tooth_tip_angle"] = 10
            problem["slot_radius_stator"] = 0.005
            problem["slot_radius_tooth"] = 0.005
            problem["shoe_spacing"] = 0.01
            problem["coolant_thickness"] = 0.01
            problem["num_slots"] = 27

            problem.run_model()

            problem.model.list_inputs(units=True, prom_name=True)
            problem.model.list_outputs(
                residuals=True, units=True, prom_name=True)

            self.assertAlmostEqual(0.018417781400324195,
                                   problem["shoe_inner_length"][0])
            self.assertAlmostEqual(0.03038482205330717,
                                   problem["tooth_length"][0])
            self.assertAlmostEqual(0.02150424858700651,
                                   problem["stator_yoke_arc_length"][0])
            self.assertAlmostEqual(0.00102041,
                                   problem["slot_area"][0])

            # problem['shoe_spacing'] = 0.005
            # problem['slot_depth'] = 0.04251922
            # problem['slot_radius_stator'] = 0.01
            # problem['slot_radius_tooth'] = 0.001
            # problem['stator_ir'] = 0.05853172
            # problem['stator_or'] = 0.10305094
            # problem['tooth_tip_angle'] = 2.52892092
            # problem['tooth_tip_thickness'] = 0.001
            # problem['tooth_width'] = 0.005
            # problem["coolant_thickness"] = 0.01
            # problem["num_slots"] = 27

            # problem.run_model()

            # problem.model.list_inputs(units=True, prom_name=True)
            # problem.model.list_outputs(
            #     residuals=True, units=True, prom_name=True)

    unittest.main()
