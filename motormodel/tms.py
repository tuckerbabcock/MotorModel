import numpy as np

import openmdao.api as om


class ThermalManagementSystem(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            "total_motor_loss", desc=" the sum of power needed to be rejected by the thermal management system")
        
        self.add_output("tms_mass")
        self.add_output("tms_power_req")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        p_rej = inputs["total_motor_loss"]

        outputs["tms_mass"] = 0.407 * (p_rej*1e-3) + 1.504
        outputs["tms_power_req"] = 0.265 * p_rej + 0.133
    

