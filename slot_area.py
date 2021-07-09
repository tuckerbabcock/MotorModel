import openmdao.api as om
import numpy as np

class SlotArea(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("num_slots", types=int)

    def setup(self):
        self.add_input("stator_inner_radius", units = "m", desc=" The inner radius of the stator")
        self.add_input("tooth_tip_thickness", units = "m", desc=" The thickness at the end of the tooth")
        self.add_input("tooth_tip_angle", units = "degrees", desc=" The angle between the flat on the back of the shoe and the horizontal")
        self.add_input("slot_depth", units = "m", desc=" The distance between the the stator inner radius and the edge of the stator yoke")
        self.add_input("tooth_width", units = "m", desc=" The width of the tooth")
        self.add_input("shoe_spacing", units = "m", desc=" The arc length distance between the tips of the stator teeth")
        
        self.add_output("area", units = "m^2", desc="The area of a winding slot")

    def compute(self, inputs, outputs):
        num_slots = self.options["num_slots"]
        sir = inputs["stator_inner_radius"]
        ttt = inputs["tooth_tip_thinkness"]
        tta = inputs["tooth_tip_angle"] * np.pi / 180 # convert to radians
        ds = inputs["slot_depth"]
        wt = inputs["tooth_width"]
        shoe_spacing = inputs["shoe_spacing"]

        shoe_spacing_angle  = shoe_spacing / sir # radians
        sa  = 2*np.pi / num_slots-shoe_spacing_angle # radians

        x = np.array([sir+ttt,
                     (sir+ttt)*np.cos(sa/2),
                     wt/2,
                     sir+ds,
                     sir+ds])
        y = np.array([0.0,
                     (sir+ttt)*np.sin(sa/2),
                     np.tan(tta - np.pi/2)*wt/2 + (sir+ttt)*(np.sin(sa/2)-np.tan(tta-np.pi/2)*np.cos(sa/2)),
                     wt/2,
                     0.0])

        tooth_area = _polygon_area(x, y) * -2 # negative since the coordinates trace half a tooth clockwise
        winding_band_area = (np.pi * (sir+ds)**2 - np.pi * (sir)**2) / num_slots
        slot_area = winding_band_area - tooth_area

        outputs["area"] =  slot_area

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


if __name__ == '__main__':
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


    unittest.main()