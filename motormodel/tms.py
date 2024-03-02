import numpy as np

import openmdao.api as om


class ThermalManagementSystem(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            "total_loss", desc=" The sum of power needed to be rejected by the thermal management system")

        self.add_output("mass")
        self.add_output("power_req")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        p_rej = inputs["total_loss"]

        outputs["mass"] = 0.407 * (p_rej*1e-3) + 1.504
        outputs["power_req"] = 0.265 * p_rej + 0.133
