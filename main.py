import json
import os
import numpy as np
import matplotlib.pyplot as plt

from solvers import run_simulation, save_csv
from physics_model import compute_theoretical_values


def load_config(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    params = config["parameters"]
    init = config["initial_conditions"]
    sim = config["simulation"]

    if "elementary_charge" not in params:
        params["elementary_charge"] = abs(params["q"])

    if "v0" in init and "theta_deg" in init:
        theta_rad = np.radians(init["theta_deg"])
        init["vx0"] = init["v0"] * np.sin(theta_rad)
        init["vy0"] = 0.0
        init["vz0"] = init["v0"] * np.cos(theta_rad)
    else:
        vx0 = init.get("vx0", 0.0)
        vy0 = init.get("vy0", 0.0)
        vz0 = init.get("vz0", 0.0)
        v0 = np.sqrt(vx0**2 + vy0**2 + vz0**2)
        init["v0"] = v0
        init["theta_deg"] = np.degrees(np.arccos(vz0 / v0)) if v0 > 0 else 90.0

    theta = init["theta_deg"]
    Ey = params.get("Ey", 0.0)
    tau = params.get("tau", 0.0)

    if tau > 0:
        config["system"] = "Hall Effect (Drude-Lorentz)"
    elif abs(theta - 90.0) < 1e-3 and abs(Ey) > 0:
        config["system"] = "Velocity Selector"
    elif abs(theta - 90.0) < 1e-3:
        config["system"] = "Cyclotron Motion"
    elif 0.0 < theta < 90.0:
        config["system"] = f"Helical Motion (theta = {theta:.1f} deg)"
    elif abs(theta) < 1e-3 or abs(theta - 180.0) < 1e-3:
        config["system"] = "Linear Motion along B"
    else:
        config["system"] = f"Electron Motion (theta = {theta:.1f} deg)"

    if params["m"] <= 0 or params["elementary_charge"] <= 0 or params["q"] == 0:
        raise ValueError("Invalid mass or charge parameters.")
    if sim["dt"] <= 0 or sim["t_max"] <= 0 or sim["dt"] >= sim["t_max"]:
        raise ValueError("Invalid time step dt or simulation duration t_max.")
    if init["v0"] < 0 or init["v0"] >= 3.0e8:
        raise ValueError("Initial speed v0 must be non-negative and sub-relativistic.")

    return config


def make_plots(results, config):
    os.makedirs("visualizations", exist_ok=True)
    params = config["parameters"]

    t = results["time_t"]
    x = results["pos_x"]
    y = results["pos_y"]
    z = results["pos_z"]
    vx = results["vel_x"]
    vy = results["vel_y"]
    vz = results["vel_z"]
    total_e = results["total_energy"]

    # 1. Trajectory / Spatial Plot (3D path and 2D xy projection)
    fig = plt.figure(figsize=(11, 4.8))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(x, y, z, color="blue", label="Trajectory")
    ax1.plot([x[0]], [y[0]], [z[0]], marker="o", color="green", label="Start")
    ax1.plot([x[-1]], [y[-1]], [z[-1]], marker="^", color="red", label="End")
    ax1.set_title("3D Electron Trajectory")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_zlabel("z (m)")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, y, color="blue", label="xy-plane")
    ax2.plot([x[0]], [y[0]], marker="o", color="green", label="Start")
    ax2.plot([x[-1]], [y[-1]], marker="^", color="red", label="End")
    ax2.set_title("XY Projection")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.axis("equal")
    ax2.grid(True, ls="--")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("visualizations/trajectory.png", dpi=300)
    plt.close()

    # 2. Time-Series of State Variables (positions and velocities)
    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)

    ax_pos.plot(t, x, color="red", label="x(t)")
    ax_pos.plot(t, y, color="blue", label="y(t)")
    ax_pos.plot(t, z, color="green", label="z(t)")
    ax_pos.set_title("Position vs. Time")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.grid(True, ls="--")
    ax_pos.legend()

    ax_vel.plot(t, vx, color="red", label="vx(t)")
    ax_vel.plot(t, vy, color="blue", label="vy(t)")
    ax_vel.plot(t, vz, color="green", label="vz(t)")
    ax_vel.set_title("Velocity vs. Time")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.grid(True, ls="--")
    ax_vel.legend()

    plt.tight_layout()
    plt.savefig("visualizations/time_series.png", dpi=300)
    plt.close()

    # 3. Phase Space Diagram: Velocity vs. Position (v(t) vs x(t))
    plt.figure(figsize=(7, 5))
    plt.plot(x, vx, color="blue", label="vx vs x")
    plt.plot(y, vy, color="green", linestyle="--", label="vy vs y")
    plt.title("Phase Space Diagram")
    plt.xlabel("Position (m)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True, ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/phase_space.png", dpi=300)
    plt.close()

    # 4. Energy Conservation and Fractional Error Plot (Guidelines Section 5)
    rel_err = np.abs((total_e - total_e[0]) / total_e[0])

    plt.figure(figsize=(8, 4))
    plt.plot(t, rel_err, color="red", label="Relative Energy Error")
    plt.yscale("log")
    plt.title("Integration Error over Time")
    plt.xlabel("Time (s)")
    plt.ylabel(r"Fractional Error ($\Delta E/E_{0}$)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/energy_error.png", dpi=300)
    plt.close()

    # 5. Hall Voltage vs. Magnetic Field Characteristic (Part 2 & 3: |V_H| = I*B / (n*e*t))
    B_vals = np.linspace(0.0, 1.0, 100)
    e = params["elementary_charge"]
    n = params["n_density"]
    t_thick = params["thickness_t"]
    I = params["current_I"]

    vh_base = (I * B_vals) / (n * e * t_thick)
    vh_2I = (2.0 * I * B_vals) / (n * e * t_thick)
    vh_half_t = (I * B_vals) / (n * e * (0.5 * t_thick))

    plt.figure(figsize=(8, 4.2))
    plt.plot(B_vals, vh_base, color="blue", label="I = 10 mA, t = 0.1 mm")
    plt.plot(B_vals, vh_2I, color="red", label="I = 20 mA, t = 0.1 mm")
    plt.plot(B_vals, vh_half_t, color="green", linestyle="--", label="I = 10 mA, t = 0.05 mm")
    plt.title("Hall Voltage vs. Magnetic Field")
    plt.xlabel("Magnetic Field B (T)")
    plt.ylabel(r"Hall Voltage $|V_H|$ (V)")
    plt.grid(True, ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/hall_voltage.png", dpi=300)
    plt.close()


def main():
    np.random.seed(42)

    config = load_config("inputs/initial_state.json")
    results = run_simulation(config)

    os.makedirs("outputs", exist_ok=True)
    save_csv(results, "outputs/simulation_results.csv")

    theory = compute_theoretical_values(config["parameters"], config["initial_conditions"])
    x = results["pos_x"]
    r_sim = 0.5 * (np.max(x) - np.min(x))

    print(f"Regime:           {config['system']}")
    print(f"Angle (deg):      {theory['theta_deg']:.2f}")
    print(f"Theory r_c (m):   {theory['r_c']:.6e}")
    print(f"Sim r_c (m):      {r_sim:.6e}")
    print(f"Omega_c (rad/s):  {theory['omega_c']:.6e}")
    print(f"Hall V_H (V):     {theory['V_H']:.6e}")
    print(f"Hall R_H (m^3/C): {theory['R_H']:.6e}")

    make_plots(results, config)


if __name__ == "__main__":
    main()
