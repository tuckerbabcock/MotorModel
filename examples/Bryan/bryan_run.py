import numpy as np
import openmdao.api as om

from sequential_motor.sequential_motor import SequentialMotor

if __name__ == "__main__":
    problem = om.Problem(name="Bryan")

    options = {
        "space-dis": {
            "degree": 1,
        },
        "external-fields": {
            "temperature": {
                "num-states": 1,
                "degree": 1,
                "basis-type": "h1"
            }
        },
        "UseCAL2forCoreLoss": True
    }
    problem.model = SequentialMotor(em_options=options)

    problem.setup(mode="rev")

    EMSolver = problem.model.analysis.coupling.em.solvers[0]
    try:
        ThermalSolver = problem.model.analysis.coupling.thermal.solver
        problem.set_val("h", 250)
        problem.set_val("fluid_temp", 293.15)
    except:
        ThermalSolver = None

    # strand_current_density = 11e6
    # strand_radius = 0.00016
    # problem.set_val("strands_in_hand", 1)
    # problem.set_val("num_turns", 504)
    # problem.set_val("rms_current_density", strand_current_density)
    # problem.set_val("strand_radius", strand_radius)
    # problem.set_val("rpm", 6000)

    problem.set_val("stack_length", 0.0418069)
    problem.set_val("strands_in_hand", 1)
    problem.set_val("num_turns", 15)
    problem.set_val("rms_current_density", 11e6)
    problem.set_val("strand_radius", 0.0007995)
    problem.set_val("rpm", 6000)

    # problem.set_val("stator_od", 0.206689)
    # problem.set_val("stator_id", 0.188171)
    # problem.set_val("rotor_od", 0.181987)
    # problem.set_val("rotor_id", 0.176308)
    # problem.set_val("slot_depth", 0.00681009)
    # problem.set_val("tooth_width", 0.00473271)
    # problem.set_val("magnet_thickness", 0.002092)
    # problem.set_val("tooth_tip_thickness", 0.000816392)
    # problem.set_val("tooth_tip_angle", 8.5)

    # Let temperature field be T(x)= T0 + T_mult*np.linalg.norm(x)**x_exp
    T0 = np.array([373.15])#, 60+273.15, 100+273.15, 140+273.15, 20+273.15, 20+273.15, 60+273.15, 100+273.15, 20+273.15, 60+273.15, 100+273.15, 140+273.15]) 
    T_mult = np.array([0])#, 0, 0, 0, 100, -100, 60, 140, 140, 60, 100, 20])
    x_exp = np.array([0])#, 0, 0, 0, 1, 1, 1, 1, 2, 3, 4, 5])
    CoreLosses = np.zeros(np.size(T0))
    for i in range(np.size(CoreLosses)):

        # Set the temperature field for the EM solver
        temp_size = EMSolver.getFieldSize("temperature")
        temperature = np.zeros(temp_size)
        temperature_func = lambda x: T0[i]+T_mult[i]*np.linalg.norm(x)**x_exp[i]
        EMSolver.setState(temperature_func, temperature, "temperature")

        try:
            # TODO: May need to be "conduct_state", but not a big deal if don't have
            ThermalSolver.setState(temperature_func, temperature, "state")
        except:
            a=1 # Do nothing (uncoupled, no thermal solver)

        problem.run_model()
        # If want to see flux magnitude, again do problem.run_model() (uncomment the below line)
        # problem.run_model()
        
        print(f"Power out: {problem.get_val('power_out')}")
        print(f"efficiency: {problem.get_val('efficiency')}")
        print(f"avg torque: {problem.get_val('average_torque')}")
        print(f"ac loss: {problem.get_val('ac_loss')}")
        print(f"dc loss: {problem.get_val('dc_loss')}")
        print(f"rms current density: {problem.get_val('rms_current_density')}")
        print(f"rms current: {problem.get_val('rms_current')}")
        print(f"stator core loss: {problem.get_val('stator_core_loss')}")
        print(f"stator max flux: {problem.get_val('max_flux_magnitude:stator')}")
        print(f"stator volume: {problem.get_val('stator_volume')}")
        print(f"stator mass: {problem.get_val('stator_mass')}")
        print(f"airgap avg flux: {problem.get_val('average_flux_magnitude:airgap')}")
        
        problem.model.list_outputs(residuals=True, units=True, prom_name=True)

        # Obtain the core losses for the current iteration
        CoreLosses[i] = problem.get_val('stator_core_loss')

    # Print the results of running the code multiple times
    print("constTempCoreLosses:")
    for i in range(np.size(CoreLosses)):
        print(f"T = {T0[i]}+{T_mult[i]}*np.linalg.norm(x)**{x_exp[i]}, Stator Core Loss = {CoreLosses[i]}")