import numpy as np

import openmdao.api as om


class ParksTransform(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("theta_e", default=0.0, desc=" Electrical angle")

    def setup(self):
        self.add_input("phaseA",
                       desc="phase A's value")
        self.add_input("phaseB",
                       desc="phase B's value")
        self.add_input("phaseC",
                       desc="phase C's value")

        self.add_output("d")
        self.add_output("q")

        self.declare_partials("*", "*", method='cs')

    def compute(self, inputs, outputs):
        theta_e = self.options["theta_e"]

        phase_a = inputs["phaseA"]
        phase_b = inputs["phaseB"]
        phase_c = inputs["phaseC"]

        abc_vec = np.array([phase_a, phase_b, phase_c])

        # a-phase to q-axis alignment
        # mat = 2/3 * np.array([
        #     [np.sin(theta_e), np.sin(theta_e - 2*np.pi / 3),
        #      np.sin(theta_e + 2*np.pi / 3)],
        #     [np.cos(theta_e), np.cos(theta_e - 2*np.pi / 3),
        #      np.cos(theta_e + 2*np.pi / 3)],
        #     [0.5, 0.5, 0.5]])

        # a-phase to d-axis alignment
        mat = 2/3 * np.array([
            [np.cos(theta_e), np.cos(theta_e - 2*np.pi / 3),
             np.cos(theta_e + 2*np.pi / 3)],
            [np.sin(theta_e), np.sin(theta_e - 2*np.pi / 3),
             np.sin(theta_e + 2*np.pi / 3)],
            [0.5, 0.5, 0.5]])

        # mat = np.sqrt(2/3) * np.array([
        #     [np.cos(theta_e), np.cos(theta_e - 2*np.pi / 3),
        #      np.cos(theta_e + 2*np.pi / 3)],
        #     [-np.sin(theta_e), -np.sin(theta_e - 2*np.pi / 3), -
        #      np.sin(theta_e + 2*np.pi / 3)],
        #     np.sqrt([0.5, 0.5, 0.5])])

        dq_vec = mat @ abc_vec
        outputs["d"] = dq_vec[0]
        outputs["q"] = dq_vec[1]


if __name__ == "__main__":
    import unittest

    class TestParksTransform(unittest.TestCase):
        def test_parks_transform_current(self):
            prob = om.Problem()

            prob.model.add_subsystem("parks",
                                     ParksTransform(
                                         theta_e=0.6690189257727805),
                                     promotes=["*"])

            prob.setup()

            prob["phaseA"] = 53.13914948
            prob["phaseB"] = -84.77403384
            prob["phaseC"] = 31.63488436

            prob.run_model()

            prob.model.list_inputs()
            prob.model.list_outputs()

        # def test_parks_transform_flux_linkage(self):
        #     prob = om.Problem()

        #     prob.model.add_subsystem("parks",
        #                              ParksTransform(
        #                                  theta_e=0.6690189257727805),
        #                              promotes=["*"])

        #     prob.setup()

        #     prob["phaseA"] = -0.00029803
        #     prob["phaseB"] = -0.00062515
        #     prob["phaseC"] = 0.00087203

        #     prob.run_model()

        #     prob.model.list_inputs()
        #     prob.model.list_outputs()

        # def test_parks_transform_flux_linkage2(self):
        #     prob = om.Problem()

        #     prob.model.add_subsystem("parks",
        #                              ParksTransform(
        #                                  theta_e=1.1926177013710793),
        #                              promotes=["*"])

        #     prob.setup()

        #     prob["phaseA"] = 0.0002251
        #     prob["phaseB"] = -0.00082999
        #     prob["phaseC"] = 0.00072155

        #     prob.run_model()

        #     prob.model.list_inputs()
        #     prob.model.list_outputs()

        # def test_parks_transform_flux_linkage3(self):
        #     prob = om.Problem()

        #     prob.model.add_subsystem("parks",
        #                              ParksTransform(
        #                                  theta_e=1.7162164769693782),
        #                              promotes=["*"])

        #     prob.setup()

        #     prob["phaseA"] = 0.00062521
        #     prob["phaseB"] = -0.00087204
        #     prob["phaseC"] = 0.000298

        #     prob.run_model()

        #     prob.model.list_inputs()
        #     prob.model.list_outputs()

    unittest.main()
