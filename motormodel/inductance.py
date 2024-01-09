import numpy as np

import openmdao.api as om


class Inductance(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("n", desc="number of distinct motor solves")

    def setup(self):
        n = self.options['n']

        self.add_input("L_le", val=0.0, desc="End-turn leakage inductance")

        for idx in range(n):
            self.add_input(f"flux_linkage_d{idx}")
            self.add_input(f"flux_linkage_q{idx}")

            self.add_input(f"current_d{idx}")
            self.add_input(f"current_q{idx}")

        self.add_output("L")
        # self.add_output("L_q")
        # self.add_output("L_d")
        # self.add_output("L_fd")

    def compute(self, inputs, outputs):
        n = self.options['n']

        L_le = inputs["L_le"]

        L_q = 0
        for idx in range(n):
            L_q += inputs[f"flux_linkage_q{idx}"] / inputs[f"current_q{idx}"]

        outputs["L"] = L_q / n + L_le

        # L_d = 0
        # for idx in range(n-1):
        #     delta_flux_linkage = inputs[f"flux_linkage_q{idx}"] - \
        #         inputs[f"flux_linkage_q{idx+1}"]
        #     delta_current = inputs[f"current_q{idx}"] - \
        #         inputs[f"current_q{idx+1}"]

        #     if np.abs(delta_current) < 1e-14:
        #         continue
        #     L_d += delta_flux_linkage / delta_current

        # outputs["L_d"] = L_d / (n-1) + L_le


if __name__ == "__main__":
    import unittest

    class TestInductance(unittest.TestCase):
        def test_inductance(self):
            prob = om.Problem()

            prob.model.add_subsystem("inductance",
                                     Inductance(n=3),
                                     promotes=['*'])

            prob.setup()

            prob.run_model()

            prob.model.list_inputs()
            prob.model.list_outputs()

    unittest.main()
