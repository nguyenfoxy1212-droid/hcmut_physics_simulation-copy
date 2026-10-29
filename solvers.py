import csv
import numpy as np
from scipy.integrate import solve_ivp

from physics_model import equations_of_motion, compute_conserved_quantities


def run_simulation(config):
    params = config["parameters"]
    init = config["initial_conditions"]
    sim = config["simulation"]

    y0 = [
        init["x0"], init["y0"], init["z0"],
        init["vx0"], init["vy0"], init["vz0"]
    ]

    dt = sim["dt"]
    t_max = sim["t_max"]
    n_steps = int(round(t_max / dt)) + 1
    t_eval = np.linspace(0.0, t_max, n_steps)

    sol = solve_ivp(
        equations_of_motion,
        (0.0, t_max),
        y0,
        method=sim["method"],
        t_eval=t_eval,
        args=(params,),
        rtol=1e-9,
        atol=1e-12
    )

    t = sol.t
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    vx, vy, vz = sol.y[3], sol.y[4], sol.y[5]

    ke, pe, total_e = compute_conserved_quantities(x, y, z, vx, vy, vz, params)

    return {
        "time_t": t,
        "pos_x": x,
        "pos_y": y,
        "pos_z": z,
        "vel_x": vx,
        "vel_y": vy,
        "vel_z": vz,
        "kinetic_energy": ke,
        "potential_energy": pe,
        "total_energy": total_e,
    }


def save_csv(results, csv_path):
    headers = list(results.keys())
    n_rows = len(results["time_t"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(n_rows):
            writer.writerow([float(results[col][i]) for col in headers])
