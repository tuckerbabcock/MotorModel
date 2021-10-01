import openmdao.api as om
import numpy as np

class SlotArea(om.ExplicitComponent):
    """
    Component that calculates the slot area based on an polygonal approximation 
    of the tooth geometry
    """
    def initialize(self):
        self.options.declare("num_slots", types=int)

    def setup(self):
        self.add_input("stator_inner_radius", desc=" The inner radius of the stator")
        self.add_input("tooth_tip_thickness", desc=" The thickness at the end of the tooth")
        self.add_input("tooth_tip_angle", desc=" The angle between the flat on the back of the shoe and the horizontal")
        self.add_input("slot_depth", desc=" The distance between the the stator inner radius and the edge of the stator yoke")
        self.add_input("slot_radius", desc=" The radius of the fillet between the tooth and stator yoke")
        self.add_input("tooth_width", desc=" The width of the tooth")
        self.add_input("shoe_spacing", desc=" The arc length distance between the tips of the stator teeth")
        
        self.add_output("slot_area", desc="The area of a winding slot")

    def compute(self, inputs, outputs):
        num_slots = self.options["num_slots"]
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

class CopperArea(om.ExplicitComponent):
    """
    Component that calculates the copper area based on the number of strands, turns,
    and wire radius
    """
    def initialize(self):
        self.options.declare("num_turns", types=int, desc=" The number of turns of wire")

    def setup(self):
        self.add_input("num_strands", desc=" Number of strands in hand for litz wire")
        self.add_input("strand_radius", desc=" Radius of one strand of litz wire")
        
        self.add_output("strand_area", desc=" The area of one strand of litz wire")
        self.add_output("copper_area", desc=" The copper area in a winding slot")

    def compute(self, inputs, outputs):
        num_turns = self.options["num_turns"]
        num_strands = inputs["num_strands"]
        strand_radius = inputs["strand_radius"]

        strand_area = np.pi * strand_radius ** 2
        outputs["strand_area"] = strand_area
        outputs["copper_area"] = strand_area * num_strands * num_turns

class MotorCurrent(om.Group):
    """
    Group that combines the SlotArea and CopperArea components to output the required
    peak current density and rms current that should be used by the high-fidelity 
    EM model that account for slot fill factor
    """

    def initialize(self):
        self.options.declare("num_turns", types=int, desc=" The number of turns of wire")
        self.options.declare("num_slots", types=int, desc=" The number of teeth in the stator")

    def setup(self):
        num_turns = self.options["num_turns"]
        num_slots = self.options["num_slots"]

        self.add_subsystem("slot_area", SlotArea(num_slots=num_slots),
                           promotes=["*"])
        self.add_subsystem("copper_area", CopperArea(num_turns=num_turns),
                           promotes=["*"])

        self.add_subsystem("fill_factor", om.ExecComp("fill_factor = copper_area / slot_area"),
                           promotes=["*"])

        # not sure why this is cube root instead of square root, but it appears to be the same as MotorCAD
        self.add_subsystem("current_density",
                           om.ExecComp("current_density = rms_current_density * power(2, 1./3) * fill_factor"),
                           promotes=["*"])

        self.add_subsystem("rms_current",
                           om.ExecComp("rms_current = rms_current_density * strand_area * num_strands"),
                           promotes=["*"])

if __name__ == "__main__":
    import unittest

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
            problem.model = MotorCurrent(num_turns=14, num_slots=24)
            problem.setup()

            problem["stator_inner_radius"] = 0.06225
            problem["tooth_tip_thickness"] = 0.001
            problem["tooth_tip_angle"] = 10
            problem["slot_depth"] = 0.01110
            problem["slot_radius"] = 0.001
            problem["tooth_width"] = 0.00430
            problem["shoe_spacing"] = 0.0035
            problem["num_strands"] = 42
            problem["strand_radius"] = 0.00016

            problem["rms_current_density"] = 11e6
            
            problem.run_model()

            self.assertAlmostEqual(8989094.151658632, problem.get_val("current_density")[0])
            self.assertAlmostEqual(0.6486044323901864, problem.get_val("fill_factor")[0])
            self.assertAlmostEqual(37.1562446325372, problem.get_val("rms_current")[0])

    unittest.main()