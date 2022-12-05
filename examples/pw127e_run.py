import openmdao.api as om

from motormodel.motors import PW127E

if __name__ == "__main__":
    problem = om.Problem(name="PW127-E", reports="n2")

    options = {
        "space-dis": {
            "degree": 1,
        }
    }
    problem.model = PW127E(em_options=options)

    problem.setup(mode="rev")

    problem.set_val("stack_length", 0.310)
    problem.set_val("strands_in_hand", 1)
    problem.set_val("num_turns", 38)
    problem.set_val("rms_current_density", 15e6)
    problem.set_val("strand_radius", 0.001671)
    problem.set_val("rpm", 1200)
    problem.set_val("rotor_od", 0.517)
    problem.set_val("magnet_thickness", 0.015)

    problem.run_model()

    print(f"Power out: {problem.get_val('power_out')}")
    print(f"efficiency: {problem.get_val('efficiency')}")
    print(f"avg torque: {problem.get_val('average_torque')}")
    print(f"mag energy: {problem.get_val('energy')}")
    print(f"ac loss: {problem.get_val('ac_loss')}")
    print(f"dc loss: {problem.get_val('dc_loss')}")
    print(f"rms current density: {problem.get_val('rms_current_density')}")
    print(f"rms current: {problem.get_val('rms_current')}")
    print(f"stator core loss: {problem.get_val('stator_core_loss')}")
    print(f"stator max flux: {problem.get_val('max_flux_magnitude:stator')}")
    print(f"stator volume: {problem.get_val('stator_volume')}")
    print(f"stator mass: {problem.get_val('stator_mass')}")
    print(f"airgap avg flux: {problem.get_val('average_flux_magnitude:airgap')}")

    print(f"state size: {problem.get_val('analysis.em_state0').shape}")

    problem.model.list_outputs(residuals=True, units=True, prom_name=True)
