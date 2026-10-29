import numpy as np


def equations_of_motion(t, state, params):
    x, y, z, vx, vy, vz = state

    v = np.array([vx, vy, vz])
    E = np.array([params["Ex"], params["Ey"], params["Ez"]])
    B = np.array([params["Bx"], params["By"], params["Bz"]])

    q = params["q"]
    m = params["m"]
    tau = params["tau"]

    F = q * (E + np.cross(v, B))
    ax, ay, az = F / m

    if tau > 0:
        ax -= vx / tau
        ay -= vy / tau
        az -= vz / tau

    return [vx, vy, vz, ax, ay, az]


def compute_conserved_quantities(x, y, z, vx, vy, vz, params):
    q = params["q"]
    m = params["m"]

    v_sq = vx**2 + vy**2 + vz**2
    ke = 0.5 * m * v_sq
    pe = -q * (params["Ex"] * x + params["Ey"] * y + params["Ez"] * z)
    total_e = ke + pe

    return ke, pe, total_e


def compute_theoretical_values(params, init):
    e = params["elementary_charge"]
    q = params["q"]
    m = params["m"]

    v0 = init["v0"]
    theta_deg = init["theta_deg"]
    theta_rad = np.radians(theta_deg)

    v_perp = v0 * np.sin(theta_rad)
    v_parallel = v0 * np.cos(theta_rad)

    E_mag = np.sqrt(params["Ex"]**2 + params["Ey"]**2 + params["Ez"]**2)
    B_mag = np.sqrt(params["Bx"]**2 + params["By"]**2 + params["Bz"]**2)

    if B_mag > 0:
        r_c = (m * v_perp) / (e * B_mag)
        omega_c = (e * B_mag) / m
        v_sel = E_mag / B_mag
    else:
        r_c = 0.0
        omega_c = 0.0
        v_sel = 0.0

    n = params["n_density"]
    w = params["width_w"]
    t_thick = params["thickness_t"]
    I = params["current_I"]

    v_d = I / (n * e * w * t_thick)
    E_H = v_d * B_mag
    V_H = (I * B_mag) / (n * e * t_thick)
    R_H = 1.0 / (n * q)
    S_B = I / (n * e * t_thick)

    return {
        "theta_deg": theta_deg,
        "v_perp": v_perp,
        "v_parallel": v_parallel,
        "r_c": r_c,
        "omega_c": omega_c,
        "v_sel": v_sel,
        "v_d": v_d,
        "E_H": E_H,
        "V_H": V_H,
        "R_H": R_H,
        "S_B": S_B,
    }
