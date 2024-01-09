import numpy as np

import openmdao.api as om

_fluids = {
    "air": {
        "temperatures": np.array([198.15, 223.15, 248.15, 258.15, 263.15, 268.15, 273.15, 278.15, 283.15, 288.15, 293.15, 298.15, 303.15, 313.15, 323.15, 333.15, 353.15, 373.15, 398.15, 423.15, 448.15, 473.15, 498.15, 573.15, 685.15, 773.15, 873.15, 973.15, 1073.15, 1173.15, 1273.15, 1373.15]),
        "dynamic_viscosity": np.array([0.00001318, 0.00001456, 0.00001588, 0.0000164, 0.00001665, 0.0000169, 0.00001715, 0.0000174, 0.00001764, 0.00001789, 0.00001813, 0.00001837, 0.0000186, 0.00001907, 0.00001953, 0.00001999, 0.00002088, 0.00002174, 0.00002279, 0.0000238, 0.00002478, 0.00002573, 0.00002666, 0.00002928, 0.00003287, 0.00003547, 0.00003825, 0.00004085, 0.00004332, 0.00004566, 0.00004788, 0.00005001]),
        "kinematic_viscosity": np.array([0.0000074, 0.00000922, 0.00001118, 0.00001201, 0.00001243, 0.00001285, 0.00001328, 0.00001372, 0.00001416, 0.00001461, 0.00001506, 0.00001552, 0.00001598, 0.00001692, 0.00001788, 0.00001886, 0.00002088, 0.00002297, 0.00002569, 0.00002851, 0.00003144, 0.00003447, 0.0000376, 0.00004754, 0.00006382, 0.00007772, 0.00009462, 0.0001126, 0.0001317, 0.0001517, 0.0001727, 0.0001946]),
        "thermal_conductivity": np.array([0.01834, 0.02041, 0.02241, 0.0232, 0.02359, 0.02397, 0.02436, 0.02474, 0.02512, 0.0255, 0.02587, 0.02624, 0.02662, 0.02735, 0.02808, 0.0288, 0.03023, 0.03162, 0.03333, 0.035, 0.03664, 0.03825, 0.03983, 0.04441, 0.05092, 0.05579, 0.06114, 0.06632, 0.07135, 0.07626, 0.08108, 0.08583]),
        "specific_heat_const_pressure": np.array([1007.27, 1005.99, 1006.0, 1006.0, 1006.0, 1006.0, 1006.0, 1006.0, 1006.0, 1006.0, 1005.98, 1005.98, 1006.05, 1006.48, 1007.31, 1008.38, 1009.64, 1011.24, 1013.81, 1016.8, 1020.57, 1024.91, 1029.64, 1045.03, 1071.36, 1092.71, 1115.3, 1135.86, 1154.4, 1170.66, 1184.72, 1196.88]),
        "specific_heat_const_volume": np.array([716.38, 716.33, 716.53, 716.76, 716.87, 716.97, 717.1, 717.25, 717.37, 717.48, 717.66, 717.9, 718.17, 718.77, 719.41, 720.09, 721.68, 723.58, 726.37, 729.63, 733.33, 737.47, 742.04, 757.94, 784.13, 805.36, 828.09, 848.61, 867.03, 883.25, 897.39, 909.68])
    },
    "water": {
        "temperatures": np.array([273.16, 283.15, 293.15, 298.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 372.75]),
        "dynamic_viscosity": np.array([0.0017914, 0.001306, 0.0010016, 0.00089, 0.0007972, 0.0006527, 0.0005465, 0.000466, 0.0004035, 0.000354, 0.0003142, 0.0002825]),
        "kinematic_viscosity": np.array([1.7918E-06, 1.3065E-06, 1.0035E-06, 8.927E-07, 8.007E-07, 6.579E-07, 5.531E-07, 0.000000474, 4.127E-07, 3.643E-07, 3.255E-07, 2.95E-07]),
        "thermal_conductivity": np.array([0.55575, 0.57864, 0.59803, 0.60659565, 0.6145, 0.62856, 0.6406, 0.65091, 0.65969, 0.66702, 0.67288, 0.67703]),
        "specific_heat_const_pressure": np.array([4219.9, 4195.5, 4184.4, 4181.6, 4180.1, 4179.6, 4181.5, 4185.1, 4190.2, 4196.9, 4205.3, 4220.0]),
        "specific_heat_const_volume": np.array([4217.4, 4191.0, 4157.0, 4137.9, 4117.5, 4073.7, 4026.4, 3976.7, 3925.2, 3872.9, 3820.4, 3770.0]),
        "density": np.array([999.84423481, 999.7, 998.21, 997.05, 995.65, 992.22, 988.04, 983.2, 977.76, 971.79, 965.31, 958.63759465])
    },
    "PGW30": {
        "temperatures": np.array([273.15, 373.15, 473.15, 573.15, 673.15]),
        "dynamic_viscosity": np.array([0.00101, 0.00101, 0.00101, 0.00101, 0.00101]),
        "kinematic_viscosity": np.array([0.000001002421692, 0.000001002421692, 0.000001002421692, 0.000001002421692, 0.000001002421692]),
        "thermal_conductivity": np.array([0.47, 0.47, 0.47, 0.47, 0.47]),
        "specific_heat_const_pressure": np.array([3960, 3960, 3960, 3960, 3960]),
        "specific_heat_const_volume": np.array([3960, 3960, 3960, 3960, 3960]),
        "density": np.array([1007.56, 1007.56, 1007.56, 1007.56, 1007.56])
    },
    "PSF5": {
        "temperatures": np.array([273.15, 373.15, 473.15]),
        "dynamic_viscosity": np.array([0.00240, 0.00240, 0.00240]),
        "kinematic_viscosity": np.array([0.000002689376961, 0.000002689376961, 0.000002689376961]),
        "thermal_conductivity": np.array([0.11, 0.11, 0.11]),
        "specific_heat_const_pressure": np.array([1710, 1710, 1710]),
        "specific_heat_const_volume": np.array([1710, 1710, 1710]),
        "density": np.array([892.40, 892.40, 892.40])
    },
    "_test": {
        "temperatures": np.array([273.15, 373.15, 473.15]),
        "dynamic_viscosity": np.array([1.0, 1.0, 1.0]),
        "kinematic_viscosity": np.array([1.0, 1.0, 1.0]),
        "thermal_conductivity": np.array([0.11, 0.11, 0.11]),
        "specific_heat_const_pressure": np.array([10, 10, 10]),
        "specific_heat_const_volume": np.array([10, 10, 10]),
        "density": np.array([892.40, 892.40, 892.40])
    },
}


class RectangularDuctCooling(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("Re_turbulent", default=3000,
                             desc=" The Reynolds number where turbulence starts")

    def setup(self):
        self.add_input("duct_length", units='m')
        self.add_input("duct_width", units='m')
        self.add_input("duct_height", units='m')
        self.add_input("fluid_velocity", units='m/s')
        self.add_input("kinematic_viscosity", units="m**2/s")
        self.add_input("specific_heat_const_pressure", units="J/(kg*K)")
        self.add_input("dynamic_viscosity", units="N*s/m**2")
        self.add_input("thermal_conductivity", units='W/(m*K)')
        self.add_input("density", units='kg/m**3')
        self.add_input("num_ducts")

        self.add_output("heat_transfer_coefficient", units='W/(K*m**2)')
        self.add_output("flow_loss", units='W')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        Re_turbulent = self.options["Re_turbulent"]

        d_l = inputs["duct_length"]
        d_w = inputs["duct_width"]
        d_h = inputs["duct_height"]

        fluid_velocity = inputs["fluid_velocity"]
        nu = inputs["kinematic_viscosity"]

        c_p = inputs["specific_heat_const_pressure"]
        mu = inputs["dynamic_viscosity"]
        kappa = inputs["thermal_conductivity"]

        n_ducts = inputs["num_ducts"]
        rho = inputs["density"]

        D_h = 2 * d_w * d_h / (d_w + d_h)

        Re = fluid_velocity * D_h / nu
        print(f"Reynolds number: {Re}")

        # f_lam = 64 / Re
        # Nu_lam = 1.051*np.log(d_h / d_w) + 2.89

        aspect_ratio = d_w / d_h
        # Eq: 5.231
        # f_lam = 24*(1 - 1.3553*aspect_ratio + 1.9467*aspect_ratio**2 - 1.7012*aspect_ratio**3 + 0.9564*aspect_ratio**4 - 0.2537*aspect_ratio**5) / Re
        # # Eq: 5.232 
        # Nu_lam = 1.051*np.log(d_h / d_w) + 2.89

        ### Custom fit of table 8.1 from Bergman "Introduction to Heat Transfer"
        f_lam_poly = np.polynomial.polynomial.Polynomial(np.array([1.0, -1.36627463, 1.98658454, -1.62994384, 0.73398257, -0.13059864]))
        Nu_lam_poly = np.polynomial.polynomial.Polynomial(np.array([1.0, -2.57380654, 4.74820955, -4.6826252, 2.39197522, -0.48852757]))
        if aspect_ratio > 1.0:
            f_lam = 96 * f_lam_poly(1/aspect_ratio) / Re
            Nu_lam = 7.54 * Nu_lam_poly(1/aspect_ratio)
        else:
            f_lam = 96 * f_lam_poly(aspect_ratio) / Re
            Nu_lam = 7.54 * Nu_lam_poly(aspect_ratio)

        Pr = c_p * mu / kappa
        print(f"Prandlt number: {Pr}")
        f_turb = (0.79*np.log(Re) - 1.64) ** -2
        Nu_turb = (f_turb / 8) * (Re - 1000) * Pr / \
            (1 + 12.7 * (f_turb / 8)**0.5 * (Pr**(2/3) - 1))
    
        sigma =  1. / (1 + np.exp(-(Re - Re_turbulent) / 100.0));
        f = (1-sigma) * f_lam + sigma * f_turb
        Nu = (1-sigma) * Nu_lam + sigma * Nu_turb

        # if (Re < Re_turbulent):
        #     f = 64 / Re
        #     Nu = 1.051*np.log(d_h / d_w) + 2.89

        # else:
        #     Pr = c_p * mu / kappa
        #     f = (0.79*np.log(Re) - 1.64) ** -2
        #     Nu = (f / 8) * (Re - 1000) * Pr / \
        #         (1 + 12.7 * (f / 8)**0.5 * (Pr**(2/3) - 1))

        outputs["heat_transfer_coefficient"] = Nu * kappa / D_h

        pressure_loss = (f * rho * fluid_velocity**2) * d_l / (2 * D_h)
        outputs['flow_loss'] = n_ducts * \
            pressure_loss * d_h * d_w * fluid_velocity


class AirgapConvection(om.ExplicitComponent):
    def setup(self):
        self.add_input("rotor_or", units='m')
        self.add_input("stator_ir", units='m')
        self.add_input("rpm", units='rpm')
        self.add_input("kinematic_viscosity", units="m**2/s")
        self.add_input("specific_heat_const_pressure", units="J/(kg*K)")
        self.add_input("dynamic_viscosity", units="N*s/m**2")
        self.add_input("thermal_conductivity", units='W/(m*K)')

        self.add_output("heat_transfer_coefficient", units='W/(K*m**2)')
        # self.add_output("Ta")
        # self.add_output("windage_loss", units='W')

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):

        rotor_or = inputs["rotor_or"]
        stator_ir = inputs["stator_ir"]
        omega = inputs["rpm"] * 2 * np.pi / 60
        nu = inputs["kinematic_viscosity"]

        c_p = inputs["specific_heat_const_pressure"]
        mu = inputs["dynamic_viscosity"]
        kappa = inputs["thermal_conductivity"]

        gap_thickness = stator_ir - rotor_or

        Re = omega * rotor_or * gap_thickness / nu

        Ta = Re * np.sqrt(gap_thickness / rotor_or)

        Pr = c_p * mu / kappa

        print(f"Prandlt number: {Pr}")
        print(f"Taylor number: {Ta}")

        nu_lam = 2
        nu_transition = 0.202 * Ta**0.63 * Pr**0.27
        nu_turb = 0.386 * Ta**0.5 * Pr**0.27

        sigma_41 =  1. / (1 + np.exp(-(Ta - 41) / 2.0));
        sigma_100 =  1. / (1 + np.exp(-(Ta - 100) / 2.0));

        Nu = (1-sigma_100) * ((1-sigma_41) * nu_lam + sigma_41 * nu_transition) \
                + sigma_100 * nu_turb

        # if Ta < 41:
        #     Nu = 2
        # elif Ta < 100:
        #     Nu = 0.202 * Ta**0.63 * Pr**0.27
        # else:
        #     Nu = 0.386 * Ta**0.5 * Pr**0.27

        outputs["heat_transfer_coefficient"] = Nu * kappa / gap_thickness
        # outputs["Ta"] = Ta


class InternalCooling(om.Group):
    def initialize(self):
        self.options.declare("coolant_fluid", default="water",
                             desc="The fluid used for active cooling")

    def setup(self):
        material_properties = self.add_subsystem("material_properties",
                                                 om.MetaModelStructuredComp(
                                                    #  method='1D-lagrange2'),
                                                     method='cubic'),
                                                 promotes_inputs=["fluid_temp"])

        fluid = self.options["coolant_fluid"]
        material_properties.add_input(
            'fluid_temp', training_data=_fluids[fluid]["temperatures"], units="K")
        material_properties.add_output(
            'dynamic_viscosity', training_data=_fluids[fluid]["dynamic_viscosity"], units="N*s/m**2")
        material_properties.add_output(
            'kinematic_viscosity', training_data=_fluids[fluid]["kinematic_viscosity"], units="m**2/s")
        material_properties.add_output(
            'thermal_conductivity', training_data=_fluids[fluid]["thermal_conductivity"], units="W/(m*K)")
        material_properties.add_output(
            'specific_heat_const_pressure', training_data=_fluids[fluid]["specific_heat_const_pressure"], units="J/(kg*K)")
        # material_properties.add_output('specific_heat_const_volume', training_data=_fluids[fluid]["specific_heat_const_volume"], units="J/(kg*K)")
        material_properties.add_output(
            'density', training_data=_fluids[fluid]["density"], units="kg/(m**3)")

        self.add_subsystem("cooling",
                           RectangularDuctCooling(),
                           promotes_inputs=["duct_length",
                                            "duct_width",
                                            "duct_height",
                                            "fluid_velocity",
                                            "num_ducts"],
                           promotes_outputs=["heat_transfer_coefficient", "flow_loss"])

        self.connect("material_properties.kinematic_viscosity",
                     "cooling.kinematic_viscosity")
        self.connect("material_properties.specific_heat_const_pressure",
                     "cooling.specific_heat_const_pressure")
        self.connect("material_properties.dynamic_viscosity",
                     "cooling.dynamic_viscosity")
        self.connect("material_properties.thermal_conductivity",
                     "cooling.thermal_conductivity")
        self.connect("material_properties.density", "cooling.density")


class AirgapCooling(om.Group):
    def initialize(self):
        self.options.declare("airgap_fluid", default="air",
                             desc="The fluid in the airgap")

    def setup(self):
        material_properties = self.add_subsystem("material_properties",
                                                 om.MetaModelStructuredComp(
                                                    #  method='1D-lagrange3'),
                                                     method='cubic'),
                                                 promotes_inputs=["fluid_temp"])

        fluid = self.options["airgap_fluid"]
        material_properties.add_input(
            'fluid_temp', training_data=_fluids[fluid]["temperatures"], units="K")
        material_properties.add_output(
            'dynamic_viscosity', training_data=_fluids[fluid]["dynamic_viscosity"], units="N*s/m**2")
        material_properties.add_output(
            'kinematic_viscosity', training_data=_fluids[fluid]["kinematic_viscosity"], units="m**2/s")
        material_properties.add_output(
            'thermal_conductivity', training_data=_fluids[fluid]["thermal_conductivity"], units="W/(m*K)")
        material_properties.add_output(
            'specific_heat_const_pressure', training_data=_fluids[fluid]["specific_heat_const_pressure"], units="J/(kg*K)")
        # material_properties.add_output('specific_heat_const_volume', training_data=_fluids[fluid]["specific_heat_const_volume"], units="J/(kg*K)")

        self.add_subsystem("cooling",
                           AirgapConvection(),
                           promotes_inputs=["rotor_or",
                                            "stator_ir",
                                            "rpm"],
                           promotes_outputs=["heat_transfer_coefficient"])

        self.connect("material_properties.kinematic_viscosity",
                     "cooling.kinematic_viscosity")
        self.connect("material_properties.specific_heat_const_pressure",
                     "cooling.specific_heat_const_pressure")
        self.connect("material_properties.dynamic_viscosity",
                     "cooling.dynamic_viscosity")
        self.connect("material_properties.thermal_conductivity",
                     "cooling.thermal_conductivity")


if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_check_partials

    class TestInternalCooling(unittest.TestCase):
        def test_internal_cooling_test_mat(self):
            problem = om.Problem()
            problem.model.add_subsystem("internal_cooling",
                                        InternalCooling(coolant_fluid="_test"),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["*"])

            problem.setup()

            problem["fluid_temp"] = 300
            problem["fluid_velocity"] = 100
            problem["duct_width"] = 0.1
            problem["duct_height"] = 0.01

            problem.run_model()

            problem.model.list_inputs(units=True, prom_name=True)
            problem.model.list_outputs(
                residuals=True, units=True, prom_name=True)

            data = problem.check_partials(form="central")
            assert_check_partials(data)

        # def test_internal_cooling_air(self):
        #     problem = om.Problem()
        #     problem.model.add_subsystem("internal_cooling",
        #                                 InternalCooling(coolant_fluid="air"),
        #                                 promotes_inputs=["*"],
        #                                 promotes_outputs=["*"])

        #     problem.setup()

        #     problem["fluid_temp"] = 300

        #     problem.run_model()

        #     problem.model.list_inputs(units=True, prom_name=True)
        #     problem.model.list_outputs(
        #         residuals=True, units=True, prom_name=True)

        #     data = problem.check_partials(form="central")
        #     assert_check_partials(data)

        def test_internal_cooling_water(self):
            problem = om.Problem()
            problem.model.add_subsystem("internal_cooling",
                                        InternalCooling(coolant_fluid="water"),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["*"])

            problem.setup()

            problem["fluid_temp"] = 291.15
            problem["fluid_velocity"] = 10
            problem["duct_width"] = 0.1
            problem["duct_height"] = 0.01

            problem.run_model()

            problem.model.list_inputs(units=True, prom_name=True)
            problem.model.list_outputs(
                residuals=True, units=True, prom_name=True)

            data = problem.check_partials(form="central")
            assert_check_partials(data)

        def plot_internal_cooling_pgw30(self):
            name = "plot_internal_cooling_pgw30"
            inputs = {
                "fluid_temp": {
                    "val": 373.15,
                    "pert": 100, 
                },
                "duct_length": {
                    "val": 0.10159632,
                    "pert": 0.05, 
                },
                "duct_width": {
                    "val": 0.00164652,
                    "pert": 0.001, 
                },
                "duct_height": {
                    "val": 0.0184367,
                    "pert": 0.01, 
                },
                "fluid_velocity": {
                    "val": 1.0,
                    "pert": 0.9, 
                },
                "num_ducts": {
                    "val": 27,
                    "pert": 10, 
                }
            }
            
            outputs = ["heat_transfer_coefficient", "flow_loss"]
            
            n = 100

            for input in inputs.keys():

                problem = om.Problem()
                problem.model.add_subsystem("internal_cooling",
                                            InternalCooling(coolant_fluid="PGW30"),
                                            promotes_inputs=["*"],
                                            promotes_outputs=["*"])

                val = inputs[input]["val"]
                pert = inputs[input]["pert"]
                # print(val)
                # print(pert)
                problem.model.add_design_var(input,
                                             lower=val-pert,
                                             upper=val+pert)

                for output in outputs:
                    problem.model.add_objective(output)

                recorder = om.SqliteRecorder(f"{name}.sql")

                # Set up DOE driver
                problem.driver = om.DOEDriver(om.FullFactorialGenerator(levels=n))

                problem.driver.add_recorder(recorder)
                problem.driver.recording_options['record_derivatives'] = True
                
                problem.setup()

                for key in inputs.keys():
                    problem[key] = inputs[key]["val"]

                problem.run_driver()

                cr = om.CaseReader(f"{name}.sql");
                cases = cr.list_cases('driver', out_stream=None)

                dvs = []
                outs = {}
                outs_wrt_dvs = {}

                for output in outputs:
                    outs[output] = []
                    outs_wrt_dvs[output] = []

                for case in cases:
                    case_outputs = cr.get_case(case).outputs
                    dvs.append(case_outputs[input])

                    derivs = cr.get_case(case).derivatives

                    for output in outputs:
                        outs[output].append(case_outputs[output])
                        outs_wrt_dvs[output].append(derivs[output, input])

                dvs = np.squeeze(dvs)
                for output in outputs:
                    outs[output] = np.squeeze(outs[output])
                    outs_wrt_dvs[output] = np.squeeze(outs_wrt_dvs[output])

                np.save(f"{name}_{input}", dvs)
                for output in outputs:
                    np.save(f"{name}_{output}_{input}", outs[output])
                    np.save(f"{name}_{output}_wrt_{input}", outs_wrt_dvs[output])

    class TestAirgapCooling(unittest.TestCase):
        def test_airgap_cooling(self):
            problem = om.Problem()
            problem.model.add_subsystem("airgap_cooling",
                                        AirgapCooling(airgap_fluid="air"),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["*"])

            problem.setup()

            problem["fluid_temp"] = 413.15 #500
            problem["rotor_or"] = 0.04796871
            problem["stator_ir"] = 0.04896871
            problem["rpm"] = 10000

            problem.run_model()

            problem.model.list_inputs(units=True, prom_name=True)
            problem.model.list_outputs(
                residuals=True, units=True, prom_name=True)

            # data = problem.check_partials(form="central")
            # assert_check_partials(data)

            data = problem.check_totals(["heat_transfer_coefficient"], [
                                        "fluid_temp", "rpm", "rotor_or", "stator_ir"], form="central")
            # assert_check_partials(data)

        def plot_airgap_cooling_air(self):
            name = "plot_airgap_cooling_air_k2"
            inputs = {
                "fluid_temp": {
                    "val": 500,
                    "pert": 200, 
                },
                "rotor_or": {
                    "val": 0.04796871,
                    "pert": 0.00075, 
                },
                "stator_ir": {
                    "val": 0.04896871,
                    "pert": 0.00075, 
                },
                "rpm": {
                    "val": 10000,
                    "pert": 100, 
                }
            }

            
            outputs = ["heat_transfer_coefficient"]
            
            n = 1000

            for input in inputs.keys():

                problem = om.Problem()
                problem.model.add_subsystem("airgap_cooling",
                                            AirgapCooling(airgap_fluid="air"),
                                            promotes_inputs=["*"],
                                            promotes_outputs=["*"])

                val = inputs[input]["val"]
                pert = inputs[input]["pert"]
                # print(val)
                # print(pert)
                problem.model.add_design_var(input,
                                             lower=val-pert,
                                             upper=val+pert)

                for output in outputs:
                    problem.model.add_objective(output)

                recorder = om.SqliteRecorder(f"{name}.sql")

                # Set up DOE driver
                problem.driver = om.DOEDriver(om.FullFactorialGenerator(levels=n))

                problem.driver.add_recorder(recorder)
                problem.driver.recording_options['record_derivatives'] = True
                
                problem.setup()

                for key in inputs.keys():
                    problem[key] = inputs[key]["val"]

                problem.run_driver()

                cr = om.CaseReader(f"{name}.sql");
                cases = cr.list_cases('driver', out_stream=None)

                dvs = []
                outs = {}
                outs_wrt_dvs = {}

                for output in outputs:
                    outs[output] = []
                    outs_wrt_dvs[output] = []

                for case in cases:
                    case_outputs = cr.get_case(case).outputs
                    dvs.append(case_outputs[input])

                    derivs = cr.get_case(case).derivatives

                    for output in outputs:
                        outs[output].append(case_outputs[output])
                        outs_wrt_dvs[output].append(derivs[output, input])

                dvs = np.squeeze(dvs)
                for output in outputs:
                    outs[output] = np.squeeze(outs[output])
                    outs_wrt_dvs[output] = np.squeeze(outs_wrt_dvs[output])

                np.save(f"{name}_{input}", dvs)
                for output in outputs:
                    np.save(f"{name}_{output}_{input}", outs[output])
                    np.save(f"{name}_{output}_wrt_{input}", outs_wrt_dvs[output])

    unittest.main()
