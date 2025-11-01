# Joseph George and Sujoge Dua
# Pd. 6
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import brentq
import cmath
from matplotlib.animation import FuncAnimation

# part 1 stuff

m_e = 511000.0
hbar = 1240.0 / (2.0 * np.pi)
V_0 = 5.0
a = 0.75

x_0 = -2 * a
x_f = 2 * a
N_points = 1000
x_interval = np.linspace(x_0, x_f, N_points)
dx = x_interval[1] - x_interval[0]

E_interval = np.linspace(0.01, V_0 - 0.001, 1000)


def V_fsw(x):
    if abs(x) <= a: return 0.0
    else: return V_0


def schrodinger(x, y, E, V):
    psi = y[0]
    dpsi_dx = y[1]
    ddpsi_dx2 = (2.0 * m_e / (hbar**2)) * (V(x) - E) * psi
    return [dpsi_dx, ddpsi_dx2]


def shoot(energy, V, slope=0.001):
    sol = solve_ivp(fun=lambda xx, yy: schrodinger(xx, yy, energy, V),
                    t_span=(x_0, x_f),
                    y0=[0.0, slope],
                    dense_output=True,
                    rtol=1e-8,
                    atol=1e-12)
    return sol.sol(x_f)[0]


def compute_wavefunc(energy, V, slope=0.001):
    sol = solve_ivp(fun=lambda xx, yy: schrodinger(xx, yy, energy, V),
                    t_span=(x_0, x_f),
                    y0=[0.0, slope],
                    dense_output=True,
                    rtol=1e-8,
                    atol=1e-12)

    psi = sol.sol(x_interval)[0]

    return psi / np.sqrt(simpson(psi**2, x_interval))


def solve(V):
    E_n = []
    prev = shoot(E_interval[0], V)
    for i in range(1, len(E_interval)):
        current = shoot(E_interval[i], V)
        if current * prev < 0.0:
            E_low = E_interval[i - 1]
            E_high = E_interval[i]
            E_bound = brentq(lambda EE: shoot(EE, V), E_low, E_high)
            E_n.append(E_bound)
            if len(E_n) >= 6: break
        prev = current
    return E_n


# part 2 stuff


def psi_0(x):
    psi = np.zeros_like(x)
    mask = (x >= -a) & (x <= a)
    psi[mask] = np.sqrt(2240 / 81) * (x[mask] + a) * (x[mask] + 0.25) * (x[mask] - a)
    return psi


def calc_coeff(psi, eigenfunc):
    coeff = []
    for phi_n in eigenfunc:
        c_n = simpson(np.conjugate(phi_n) * psi, x_interval)
        coeff.append(c_n)
    return np.array(coeff)


def reconstruct_wavefunc(coeff, eigenfunc):
    psi_reconstructed = np.zeros(len(x_interval), dtype=complex)
    for c_n, phi_n in zip(coeff, eigenfunc):
        psi_reconstructed += c_n * phi_n
    return psi_reconstructed


def time_evolve(t, coeff, energies, eigenfunc):
    psi_t = np.zeros(len(x_interval), dtype=complex)
    for c_n, E_n, phi_n in zip(coeff, energies, eigenfunc):
        phase = cmath.exp(-1j * E_n * t / hbar)
        psi_t += c_n * phase * phi_n
    return psi_t


def x_expectation(psi):
    prob_density = np.abs(psi)**2
    prob_density /= simpson(prob_density, x_interval)
    return simpson(x_interval * prob_density, x_interval)


def main():
    energies = solve(V_fsw)
    print(f"{len(energies)} bound states:")
    for i, E in enumerate(energies):
        print(f"E_{i + 1} = {E:.6f} eV")
    print()

    eigenfunc = [compute_wavefunc(E, V_fsw) for E in energies]

    psi = psi_0(x_interval)
    psi /= np.sqrt(simpson(psi**2, x_interval))

    coeff = calc_coeff(psi, eigenfunc)
    print("c_n:")
    for i, c in enumerate(coeff):
        print(f"c_{i + 1} = {c:.6f}")
    print()

    psi_combo = reconstruct_wavefunc(coeff, eigenfunc)

    plt.figure(figsize=(10, 6))
    plt.plot(x_interval, psi, "b-", label="Initial wavefunction")
    plt.plot(x_interval, np.real(psi_combo), "r-", label="Decomposition")
    plt.axvline(-a, color="k", linestyle="-")
    plt.axvline(a, color="k", linestyle="-")
    plt.legend()
    plt.xlabel("x (nm)")
    plt.ylabel("ψ(x)")
    plt.title("Initial Wavefunction Decomposition")
    plt.savefig("wavefunc_decomposition.png")
    plt.show()

    tau = 2 * np.pi * hbar / (energies[1] - energies[0])
    print(f"τ = {tau:.6f}")

    num_frames = 240
    times = np.linspace(0, tau, num_frames)
    tau_ticks = np.linspace(0.1 * tau, 0.9 * tau, 9)

    prob_densities = []
    x_exps = []

    for t in times:
        psi_t = time_evolve(t, coeff, energies, eigenfunc)
        prob_t = np.abs(psi_t)**2
        prob_t /= simpson(prob_t, x_interval)

        prob_densities.append(prob_t)
        x_exps.append(x_expectation(psi_t))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

    line, = ax1.plot([], [], "b-", lw=2)
    x_exp_mark, = ax1.plot([], [], "ro", markersize=5)

    time_line, = ax2.plot([], [], "g-", lw=2)
    time_marker, = ax2.plot([], [], "ro", markersize=5)

    ax1.axvline(-a, color="k", linestyle="-")
    ax1.axvline(a, color="k", linestyle="-")

    ax1.set_xlim(x_0, x_f)
    max_prob = max([max(prob) for prob in prob_densities])
    ax1.set_ylim(0, max_prob * 1.1)
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("|ψ(x,t)|²")

    ax2.set_xlim(0, 1)
    y_min = min(x_exps) - 0.1 * abs(min(x_exps))
    y_max = max(x_exps) + 0.1 * abs(max(x_exps))
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel("Time (τ)")
    ax2.set_ylabel("<x> (nm)")

    def init():
        line.set_data([], [])
        x_exp_mark.set_data([], [])
        time_line.set_data([], [])
        time_marker.set_data([], [])
        return line, x_exp_mark, time_line, time_marker

    def animate(i):
        line.set_data(x_interval, prob_densities[i])
        x_exp_mark.set_data([x_exps[i]], [0.05 * max_prob])

        t_vals = times[:i + 1] / tau
        x_vals = x_exps[:i + 1]
        time_line.set_data(t_vals, x_vals)
        time_marker.set_data([times[i] / tau], [x_exps[i]])

        return line, x_exp_mark, time_line, time_marker

    ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init, blit=True, interval=16)

    ani.save("time_evolution.gif", writer="ffmpeg", fps=60)
    plt.show()

    print("\ntime\t<x>")
    for i, t in enumerate(tau_ticks):
        j = np.argmin(np.abs(times - t))
        print(f"{t / tau:.1f}τ\t{x_exps[j]:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(times / tau, x_exps, "b-")
    plt.xlabel("Time (τ)")
    plt.ylabel("<x> (nm)")
    plt.title("Expectation Value of Position vs. Time")
    plt.savefig("x_expectation_val.png")
    plt.show()


if __name__ == "__main__":
    main()
