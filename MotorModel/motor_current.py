import openmdao.api as om
import numpy as np

def _polygon_area(x, y):
    """
    Function to compute the area of a polygon by giving its coordinates in
    counter-clockwise order.
    """
    area = 0
    j = len(x)-1; 
    for i in range (0, len(x)):
        area += (x[j]+x[i]) * (y[i]-y[j]);
        j = i
    return area/2;

class SlotArea(om.ExplicitComponent):
    """
    Component that calculates the slot area based on an polygonal approximation 
    of the tooth geometry
    """
    def setup(self):
        self.add_input("num_slots",
                       desc=" The number of slots in the motor")
        self.add_input("stator_inner_radius",
                       desc=" The inner radius of the stator")
        self.add_input("tooth_tip_thickness",
                       desc=" The thickness at the end of the tooth")
        self.add_input("tooth_tip_angle",
                       desc=" The angle between the flat on the back of the shoe and the horizontal")
        self.add_input("slot_depth",
                       desc=" The distance between the the stator inner radius and the edge of the stator yoke")
        self.add_input("slot_radius",
                       desc=" The radius of the fillet between the tooth and stator yoke")
        self.add_input("tooth_width",
                       desc=" The width of the tooth")
        self.add_input("shoe_spacing",
                       desc=" The arc length distance between the tips of the stator teeth")
        
        self.add_output("slot_area",
                        desc=" The area of a winding slot")

    def setup_partials(self):
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        num_slots = inputs["num_slots"][0]
        sir = inputs["stator_inner_radius"][0]
        ttt = inputs["tooth_tip_thickness"][0]
        tta = inputs["tooth_tip_angle"][0] * np.pi / 180 # convert to radians
        ds = inputs["slot_depth"][0]
        sr = inputs["slot_radius"][0]
        wt = inputs["tooth_width"][0]
        shoe_spacing = inputs["shoe_spacing"][0]

        shoe_spacing_angle  = shoe_spacing / sir # radians
        sa  = 2*np.pi/num_slots - shoe_spacing_angle # radians

        x = np.array([sir+ttt/3,
                    (sir+ttt)*np.cos(sa/2),
                    (wt/2 - (sir+ttt)*(np.sin(sa/2)-np.tan(tta-np.pi/2)*np.cos(sa/2)))/ np.tan(tta-np.pi/2),
                    sir+ds,
                    sir+ds])
        y = np.array([0.0,
                    (sir+ttt)*np.sin(sa/2),
                    wt/2.0,
                    wt/2.0,
                    0.0])
    
        tooth_area = _polygon_area(x, y) * -1 # negative since the coordinates trace half a tooth clockwise
        tooth_area += (4.0 - np.pi) / 2.0 * sr ** 2 # add tooth area for fillets
        
        winding_band_area = (np.pi*(sir+ds)**2 - np.pi*(sir)**2) / (num_slots*2)
        slot_area = winding_band_area - tooth_area
        # print("slot area: ", slot_area, "should be: ", 0.0000729, "ratio: ", slot_area / 0.0000729)
        # print("tooth area should be: ", winding_band_area - 0.0000729, "is: ", tooth_area, "ratio: ", tooth_area / (winding_band_area - 0.0000729))
        outputs["slot_area"] = slot_area

class CopperArea(om.ExplicitComponent):
    """
    Component that calculates the copper area based on the number of strands, turns,
    and wire radius
    """
    def setup(self):
        self.add_input("num_turns",
                       desc=" The number of turns of wire")
        self.add_input("strands_in_hand",
                       desc=" Number of strands in hand for litz wire")
        self.add_input("strand_radius",
                       desc=" Radius of one strand of litz wire")
        
        self.add_output("strand_area",
                        desc=" The area of one strand of litz wire")
        self.add_output("copper_area",
                        desc=" The copper area in a winding slot")

    def setup_partials(self):
        self.declare_partials("strand_area", "strand_radius")
        self.declare_partials("copper_area", "*")

    def compute(self, inputs, outputs):
        num_turns = inputs["num_turns"]
        strands_in_hand = inputs["strands_in_hand"]
        strand_radius = inputs["strand_radius"]

        strand_area = np.pi * strand_radius ** 2
        outputs["strand_area"] = strand_area
        outputs["copper_area"] = strand_area * strands_in_hand * num_turns

    def compute_partials(self, inputs, partials):
        num_turns = inputs["num_turns"]
        strands_in_hand = inputs["strands_in_hand"]
        strand_radius = inputs["strand_radius"]

        partials["strand_area", "strand_radius"] = 2 * np.pi * strand_radius

        strand_area = np.pi * strand_radius ** 2
        partials["copper_area", "num_turns"] = strand_area * strands_in_hand
        partials["copper_area", "strands_in_hand"] = strand_area * num_turns
        partials["copper_area", "strand_radius"] = 2 * np.pi * strand_radius * strands_in_hand * num_turns

class ThreePhaseCurrent(om.ExplicitComponent):
    """
    Component that maps from a peak volumetric current density to the current density
    for each phase based on electrical angle input
    """
    def initialize(self):
        self.options.declare("theta_e", default=0.0, desc=" Electrical angle")

    def setup(self):
        self.add_input("current_density",
                       desc=" Volumetric peak current density")

        self.add_output("current_density:phaseA",
                        desc=" Volumetric current density for phase A")
        self.add_output("current_density:phaseB",
                        desc=" Volumetric current density for phase B")
        self.add_output("current_density:phaseC",
                        desc=" Volumetric current density for phase C")

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        current_density = inputs["current_density"]
        theta_e = self.options["theta_e"]

        outputs["current_density:phaseA"] = current_density * np.sin(-theta_e)
        outputs["current_density:phaseB"] = current_density * np.sin(-theta_e + 2*np.pi / 3)
        outputs["current_density:phaseC"] = current_density * np.sin(-theta_e + 4*np.pi / 3)

    def compute_partials(self, inputs, partials):
        theta_e = self.options["theta_e"]

        partials["current_density:phaseA", "current_density"] = np.sin(-theta_e)
        partials["current_density:phaseB", "current_density"] = np.sin(-theta_e + 2*np.pi / 3)
        partials["current_density:phaseC", "current_density"] = np.sin(-theta_e + 4*np.pi / 3)

class MotorCurrent(om.Group):
    """
    Group that combines the SlotArea and CopperArea components to output the required
    peak current density and rms current that should be used by the high-fidelity 
    EM model that account for slot fill factor
    """

    def initialize(self):
        self.options.declare("theta_e", default=0.0, types=(float, list), desc=" Electrical angle")

    def setup(self):
        theta_e = self.options["theta_e"]

        self.add_subsystem("slot_area",
                           SlotArea(),
                           promotes_inputs=["*"],
                           promotes_outputs=["slot_area"])
        self.add_subsystem("copper_area",
                           CopperArea(),
                           promotes_inputs=["*"])

        self.add_subsystem("rms_current",
                           om.ExecComp("rms_current = rms_current_density * strand_area * strands_in_hand"),
                           promotes_inputs=["rms_current_density", "strands_in_hand"],
                           promotes_outputs=["rms_current"])
        self.connect("copper_area.strand_area", "rms_current.strand_area")

        self.add_subsystem("fill_factor", om.ExecComp("fill_factor = copper_area / slot_area"))
        self.connect("slot_area", "fill_factor.slot_area")
        self.connect("copper_area.copper_area", "fill_factor.copper_area")

        self.add_subsystem("current_density",
                           om.ExecComp("current_density = rms_current_density * power(2, 1./2) * fill_factor"),
                           promotes_inputs=["rms_current_density"],
                           promotes_outputs=["current_density"])
        self.connect("fill_factor.fill_factor", "current_density.fill_factor")

        if isinstance(theta_e, list):
            three_phase = self.add_subsystem("three_phase", om.Group())
            for idx, angle in enumerate(theta_e):
                three_phase.add_subsystem(f"three_phase{idx}",
                                          ThreePhaseCurrent(theta_e=angle),
                                          promotes_inputs=["current_density"])

            self.promotes("three_phase",
                          inputs=["current_density"],
                          outputs=["three_phase*.current_density:phase*"])
        else:
            self.add_subsystem("three_phase",
                               ThreePhaseCurrent(theta_e=theta_e),
                               promotes_inputs=["current_density"])

if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_check_partials

    class TestPolygonArea(unittest.TestCase):
        def test_square_area(self):
            x = np.array([0.0, 1.0, 1.0, 0.0])
            y = np.array([0.0, 0.0, 1.0, 1.0])

            self.assertAlmostEqual(1.0, _polygon_area(x, y))

        def test_trap_area(self):
            x = np.array([0.0, 6.0, 4.0, 1.0])
            y = np.array([0.0, 0.0, 2.0, 2.0])

            self.assertAlmostEqual(9.0, _polygon_area(x, y))

    class TestMotorCurrent(unittest.TestCase):
        def test_ref_motor_current(self):
            problem = om.Problem()
            problem.model.add_subsystem("current",
                                        MotorCurrent(),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["slot_area",
                                                          "rms_current",
                                                          "three_phase*.current_density:phase*"])

            problem.setup()

            problem["strands_in_hand"] = 42
            problem["num_turns"] = 14
            problem["num_slots"] = 24
            problem["stator_inner_radius"] = 0.06225
            problem["tooth_tip_thickness"] = 0.001
            problem["tooth_tip_angle"] = 10
            problem["slot_depth"] = 0.01110
            problem["slot_radius"] = 0.001
            problem["tooth_width"] = 0.00430
            problem["shoe_spacing"] = 0.0035
            problem["strand_radius"] = 0.00016

            problem["rms_current_density"] = 11e6
            
            problem.run_model()

            self.assertAlmostEqual(8738124.47344907, problem.get_val("three_phase.current_density:phaseB")[0])
            # self.assertAlmostEqual(0.6486044323901864, problem.get_val("fill_factor")[0])
            self.assertAlmostEqual(37.1562446325372, problem.get_val("rms_current")[0])

        def test_ref_motor_current_partials(self):
            problem = om.Problem()
            problem.model.add_subsystem("current",
                                        MotorCurrent(),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["slot_area",
                                                          "rms_current",
                                                          "three_phase*.current_density:phase*"])
            problem.setup()

            problem["strands_in_hand"] = 42
            problem["num_turns"] = 14
            problem["num_slots"] = 24
            problem["stator_inner_radius"] = 0.06225
            problem["tooth_tip_thickness"] = 0.001
            problem["tooth_tip_angle"] = 10
            problem["slot_depth"] = 0.01110
            problem["slot_radius"] = 0.001
            problem["tooth_width"] = 0.00430
            problem["shoe_spacing"] = 0.0035
            problem["strand_radius"] = 0.00016

            # relatively small current density for better scaled
            # finite difference check
            problem["rms_current_density"] = 1
            
            problem.run_model()

            data = problem.check_partials(form="central")
            assert_check_partials(data)

    unittest.main()
