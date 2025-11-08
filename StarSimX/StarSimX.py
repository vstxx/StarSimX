#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Star Formation N-body Simulator (educational)
---------------------------------------------
A minimal self-gravitating N-body simulator using leapfrog integration with Plummer softening.
Designed for classroom / IB projects to explore "formation of stars" via gravitational collapse analog.

Features
- Adjustable parameters via CLI (or config JSON).
- Initial conditions: uniform sphere or Gaussian, with optional solid-body rotation and velocity dispersion (vrms).
- Leapfrog integrator, energy logging.
- Simple diagnostics: maximum density proxy (1/min nn-distance), collapse time estimate.
- Saves static plots and an animated GIF (optional).
- GUI mode with live 2D scatter plot and controls.

Limitations
- O(N^2) force calculation: keep N <= 2000 for reasonable speed.
- No real hydrodynamics/pressure/radiative feedback - it's an N-body toy model.

Usage (examples)
----------------
GUI (default if no args):
  python StarSimX/StarSimX.py
CLI:
  python StarSimX/StarSimX.py --N 800 --vrms 0.1 --omega 0.3 --t_end 5.0 --dt 0.002 --gif out.gif
  python StarSimX/StarSimX.py --config config.json
(See README for details.)
"""
import argparse
import json
import sys
import threading
import time
from typing import Optional, Callable

import numpy as np

# Optional acceleration with numba
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def set_seed(seed: Optional[int]):
    if seed is not None:
        np.random.seed(seed)


def init_positions(N: int, radius: float, mode: str = "uniform_sphere") -> np.ndarray:
    if mode == "gaussian":
        x = np.random.normal(scale=radius / 2, size=(N, 3))
        return x
    # default: uniform sphere
    # Marsaglia method
    u = np.random.uniform(-1, 1, size=N)
    theta = np.random.uniform(0, 2 * np.pi, size=N)
    r = np.cbrt(np.random.uniform(0, 1, size=N)) * radius
    s = np.sqrt(1 - u ** 2)
    pos = np.column_stack((r * s * np.cos(theta), r * s * np.sin(theta), r * u))
    return pos


def add_rotation(pos: np.ndarray, omega: float) -> np.ndarray:
    # solid-body rotation about z-axis: v = omega x r
    x, y = pos[:, 0], pos[:, 1]
    vx = -omega * y
    vy = omega * x
    vz = np.zeros_like(vx)
    return np.column_stack((vx, vy, vz))


def add_velocity_dispersion(N: int, vrms: float) -> np.ndarray:
    # 3D Maxwellian with given RMS speed
    if vrms <= 0:
        return np.zeros((N, 3))
    v = np.random.normal(0, 1, size=(N, 3))
    # scale to target vrms
    current_rms = np.sqrt(np.mean(np.sum(v ** 2, axis=1)))
    if current_rms == 0:
        return v
    return v * (vrms / current_rms)


def compute_acc_python(pos: np.ndarray, mass: float, eps: float, G: float = 1.0) -> np.ndarray:
    # naive O(N^2)
    N = pos.shape[0]
    acc = np.zeros_like(pos)
    for i in range(N):
        r = pos[i] - pos
        dist2 = np.sum(r * r, axis=1) + eps * eps
        inv_r3 = (dist2 ** -1.5)
        inv_r3[i] = 0.0
        acc[i] = -G * mass * np.sum(r * inv_r3[:, None], axis=0)
    return acc

if _HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _compute_acc_numba(pos, mass, eps, G=1.0):
        N = pos.shape[0]
        acc = np.zeros_like(pos)
        for i in prange(N):
            xi = pos[i, 0]; yi = pos[i, 1]; zi = pos[i, 2]
            ax = 0.0; ay = 0.0; az = 0.0
            for j in range(N):
                if i == j:
                    continue
                rx = xi - pos[j, 0]
                ry = yi - pos[j, 1]
                rz = zi - pos[j, 2]
                dist2 = rx*rx + ry*ry + rz*rz + eps*eps
                invr3 = 1.0 / (dist2 ** 1.5)
                ax += rx * invr3
                ay += ry * invr3
                az += rz * invr3
            acc[i, 0] = -G * mass * ax
            acc[i, 1] = -G * mass * ay
            acc[i, 2] = -G * mass * az
        return acc

def compute_acc(pos: np.ndarray, mass: float, eps: float, G: float = 1.0) -> np.ndarray:
    """Wrapper that uses Numba if available for speed."""
    if _HAS_NUMBA:
        # Ensure contiguous float64 for numba
        p = np.ascontiguousarray(pos, dtype=np.float64)
        return _compute_acc_numba(p, mass, eps, G)
    return compute_acc_python(pos, mass, eps, G)


def kinetic_energy(vel: np.ndarray, m: float) -> float:
    return 0.5 * m * np.sum(vel * vel)


def potential_energy(pos: np.ndarray, m: float, eps: float, G: float = 1.0) -> float:
    # pairwise potential with softening ~ -G m^2 / sqrt(r^2 + eps^2)
    N = pos.shape[0]
    pe = 0.0
    for i in range(N):
        r = pos[i] - pos[i + 1 :]
        dist = np.sqrt(np.sum(r * r, axis=1) + eps * eps)
        pe += -G * m * m * np.sum(1.0 / dist)
    return pe


def nn_density_proxy(pos: np.ndarray, k: int = 8) -> float:
    # density proxy: inverse of mean distance to k nearest neighbors (simple, fast O(N^2) for small N)
    from heapq import nsmallest

    N = pos.shape[0]
    dists = []
    for i in range(N):
        di = np.sqrt(np.sum((pos[i] - pos) ** 2, axis=1))
        di = di[di > 0]
        kmin = nsmallest(k, di.tolist())
        dists.append(np.mean(kmin))
    return 1.0 / (np.mean(dists) + 1e-9)


def simulate(
    N=1000,
    mass_total=1.0,
    radius=1.0,
    eps=0.02,
    dt=0.002,
    t_end=5.0,
    vrms=0.1,
    omega=0.0,
    seed=None,
    init_mode="uniform_sphere",
    save_gif=None,
    gif_stride=5,
    out_prefix="run",
    energy_stride=10,
    step_callback: Optional[Callable[[float, int, np.ndarray, np.ndarray], None]] = None,
    callback_stride: int = 1,
    stop_event: Optional[threading.Event] = None,
    headless: bool = True,
):
    # For non-interactive plotting and GIF generation in CLI mode
    if headless:
        try:
            import matplotlib
            # Only set backend if no backend has been imported yet
            if 'matplotlib.backends' not in sys.modules and 'matplotlib.pyplot' not in sys.modules:
                matplotlib.use("Agg")
        except Exception:
            # Fail-safe: ignore backend switch issues in GUI context
            pass
    from matplotlib import pyplot as plt
    from matplotlib.animation import PillowWriter
    # Ensure Matplotlib uses Poppins (with fallbacks) for a clean look
    try:
        from matplotlib import rcParams
        rcParams['font.family'] = 'Poppins'
        rcParams['font.sans-serif'] = ['Poppins', 'Segoe UI', 'Arial', 'DejaVu Sans', 'sans-serif']
    except Exception:
        pass

    set_seed(seed)
    m = mass_total / N

    # state (can be downcast to float32 to reduce RAM if needed)
    pos = init_positions(N, radius, init_mode)
    vel = add_rotation(pos, omega) + add_velocity_dispersion(N, vrms)

    # Center-of-mass frame
    vel -= np.mean(vel, axis=0, keepdims=True)
    pos -= np.mean(pos, axis=0, keepdims=True)

    # initial acceleration
    acc = compute_acc(pos, m, eps)

    T_list, K_list, U_list, D_list = [], [], [], []

    # optional streaming GIF setup (avoid storing frames in memory)
    gif_ctx = None
    if save_gif is not None:
        fig_g = plt.figure(figsize=(5, 5))
        ax_g = fig_g.add_subplot(111)
        scat_g = ax_g.scatter([], [], s=2)
        ax_g.set_aspect("equal")
        ax_g.set_title("Collapse (x-y)")
        lim0 = max(radius * 1.2, 1e-6)
        ax_g.set_xlim(-lim0, lim0)
        ax_g.set_ylim(-lim0, lim0)
        writer = PillowWriter(fps=20)
        gif_ctx = writer.saving(fig_g, save_gif, dpi=100)
        gif_ctx.__enter__()
        # draw initial
        scat_g.set_offsets(pos[:, :2])
        writer.grab_frame()

    def write_gif_frame():
        if gif_ctx is not None:
            # update current positions
            scat_g.set_offsets(pos[:, :2])
            writer.grab_frame()

    # initial energy and density
    T_list.append(0.0)
    K_list.append(kinetic_energy(vel, m))
    U_list.append(potential_energy(pos, m, eps))
    D_list.append(nn_density_proxy(pos))
    if save_gif is not None:
        write_gif_frame()

    t = 0.0
    step = 0

    # initial callback
    if step_callback is not None:
        step_callback(t, step, pos, vel)

    # Leapfrog
    while t < t_end:
        if stop_event is not None and stop_event.is_set():
            break

        # kick (half)
        vel += 0.5 * dt * acc
        # drift
        pos += dt * vel
        # recompute acc
        acc = compute_acc(pos, m, eps)
        # kick (half)
        vel += 0.5 * dt * acc

        t += dt
        step += 1

        if step % energy_stride == 0:
            T_list.append(t)
            K_list.append(kinetic_energy(vel, m))
            U_list.append(potential_energy(pos, m, eps))
            D_list.append(nn_density_proxy(pos))

        if step_callback is not None and (step % callback_stride == 0):
            step_callback(t, step, pos, vel)

        if save_gif is not None and (step % gif_stride == 0):
            write_gif_frame()

    # finalize GIF
    if gif_ctx is not None:
        try:
            gif_ctx.__exit__(None, None, None)
        except Exception:
            pass

    # Save diagnostics
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=2)
    ax1.set_title("Final positions")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(T_list, K_list, label="Kinetic")
    ax2.plot(T_list, U_list, label="Potential")
    ax2.plot(T_list, np.array(K_list) + np.array(U_list), label="Total E")
    ax2.set_xlabel("time")
    ax2.set_ylabel("Energy (arb.)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_summary.png", dpi=150)
    plt.close(fig)

    # Density proxy plot
    fig2 = plt.figure()
    axd = fig2.add_subplot(111)
    axd.plot(T_list, D_list)
    axd.set_xlabel("time")
    axd.set_ylabel("1 / <d_kNN>")
    axd.set_title("Density proxy over time")
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_density.png", dpi=150)
    plt.close(fig)

    # Persist analytics data for the Analytics tab
    try:
        np.savez(
            f"{out_prefix}_data.npz",
            t=np.array(T_list, dtype=float),
            K=np.array(K_list, dtype=float),
            U=np.array(U_list, dtype=float),
            D=np.array(D_list, dtype=float),
        )
    except Exception:
        pass


# --- GUI helpers ------------------------------------------------------------
class _Tooltip:
    def __init__(self, widget, text: str, delay_ms: int = 600):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _):
        self._cancel()
        self._id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._id is not None:
            self.widget.after_cancel(self._id)
            self._id = None

    def _show(self):
        import tkinter as tk
        if self._tip is not None:
            return
        x, y, cx, cy = self.widget.bbox("insert") if self.widget.winfo_ismapped() else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 20
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self._tip, text=self.text, justify=tk.LEFT, relief=tk.SOLID, borderwidth=1,
                       background="#ffffe0", foreground="#000", padx=6, pady=3)
        lbl.pack()

    def _hide(self, _):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def _add_tooltip(widget, text: str):
    _Tooltip(widget, text)


# --- GUI --------------------------------------------------------------------

def launch_gui():
    import tkinter as tk
    # Prefer ttkbootstrap for a modern look; fall back to ttk
    try:
        import ttkbootstrap as tb
        ttkmod = tb
        use_bootstrap = True
    except Exception:
        import tkinter.ttk as ttkmod  # type: ignore
        use_bootstrap = False
    from tkinter import filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from PIL import Image, ImageTk
    import os
    # Matplotlib font family to keep figures consistent with UI
    try:
        from matplotlib import rcParams as _rc
        _rc['font.family'] = 'Poppins'
        _rc['font.sans-serif'] = ['Poppins', 'Segoe UI', 'Arial', 'DejaVu Sans', 'sans-serif']
    except Exception:
        pass

    # Window
    if use_bootstrap:
        root = ttkmod.Window(themename="flatly")
    else:
        root = tk.Tk()
    root.title("StarSimX - N-body Star Formation (Educational)")
    root.geometry("1200x800")

    # Global font configuration (prefer Poppins)
    try:
        import tkinter.font as tkfont
        fams = {f.lower() for f in tkfont.families(root)}
        preferred = "Poppins"
        fallback = "Segoe UI" if "segoe ui" in fams else "Arial"
        APP_FONT_FAMILY = preferred if preferred.lower() in fams else fallback
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkFixedFont", "TkHeadingFont", "TkIconFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(name).configure(family=APP_FONT_FAMILY)
            except Exception:
                pass
    except Exception:
        APP_FONT_FAMILY = "Poppins"  # best effort

    # Convenience font tuples
    FONT_BOLD_XL = (APP_FONT_FAMILY, 18, "bold")
    FONT_BOLD_LG = (APP_FONT_FAMILY, 16, "bold")
    FONT_BOLD_MD = (APP_FONT_FAMILY, 14, "bold")
    FONT_MD = (APP_FONT_FAMILY, 12)
    FONT_SM = (APP_FONT_FAMILY, 10)

    # Colors and styles
    muted = "#6c757d" if use_bootstrap else "#6b7280"

    # Layout: sidebar (col 0), main (col 1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # Sidebar
    sidebar = ttkmod.Frame(root, padding=10)
    sidebar.grid(row=0, column=0, sticky="nsw")
    sidebar.grid_rowconfigure(99, weight=1)
    if use_bootstrap:
        sidebar.configure(bootstyle="secondary")

    brand = ttkmod.Label(sidebar, text="StarSimX", font=FONT_BOLD_XL)
    brand.grid(row=0, column=0, sticky="w", pady=(0, 10))

    # Main container
    main = ttkmod.Frame(root, padding=(10, 10, 10, 10))
    main.grid(row=0, column=1, sticky="nsew")
    main.grid_rowconfigure(1, weight=1)
    main.grid_columnconfigure(0, weight=1)

    # Top bar
    topbar = ttkmod.Frame(main)
    topbar.grid(row=0, column=0, sticky="ew")
    title = ttkmod.Label(topbar, text="Dashboard", font=FONT_BOLD_LG)
    title.pack(side=tk.LEFT)
    # Dark mode toggle
    dark_var = tk.BooleanVar(value=False)
    def toggle_dark():
        val = dark_var.get()
        if use_bootstrap:
            try:
                theme = "darkly" if val else "flatly"
                root.style.theme_use(theme)  # type: ignore[attr-defined]
            except Exception:
                pass
        # apply to matplotlib axes
        def style_axes(ax, dark):
            fg = "#e5e7eb" if dark else "#111827"
            bg = "#0f172a" if dark else "#ffffff"
            ax.set_facecolor(bg)
            ax.figure.set_facecolor(bg)
            for spine in ax.spines.values():
                spine.set_color(fg)
            ax.tick_params(colors=fg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(bg)
                leg.get_frame().set_edgecolor(fg)
                for txt in leg.get_texts():
                    txt.set_color(fg)
        try:
            style_axes(ax, val)
        except Exception:
            pass
        try:
            style_axes(ax_e, val); style_axes(ax_d, val)
        except Exception:
            pass
        try:
            canvas.draw_idle()
        except Exception:
            pass
        try:
            canvas_e.draw_idle(); canvas_d.draw_idle()
        except Exception:
            pass
    ttkmod.Checkbutton(topbar, text="Dark", variable=dark_var, command=toggle_dark).pack(side=tk.RIGHT, padx=(0,8))
    search_var = tk.StringVar()
    search = ttkmod.Entry(topbar, textvariable=search_var, width=30)
    search.pack(side=tk.RIGHT)

    # Content area (pages)
    content = ttkmod.Frame(main)
    content.grid(row=1, column=0, sticky="nsew")
    content.grid_rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=1)

    pages = {}

    # Dashboard page
    page_dashboard = ttkmod.Frame(content)
    page_dashboard.grid(row=0, column=0, sticky="nsew")
    pages["Dashboard"] = page_dashboard

    dash_cards = ttkmod.Frame(page_dashboard)
    dash_cards.pack(fill=tk.X, pady=(10, 10))
    for c in range(4):
        dash_cards.grid_columnconfigure(c, weight=1)

    def make_card(parent, title_txt, value_txt="-"):
        frame = ttkmod.Labelframe(parent, text=title_txt, padding=(10, 6))
        if use_bootstrap:
            frame.configure(bootstyle="secondary")
        v = ttkmod.Label(frame, text=value_txt, font=(APP_FONT_FAMILY, 14, "bold"))
        v.pack(anchor="w")
        return frame, v

    card_particles, lbl_particles = make_card(dash_cards, "Particles")
    card_dt, lbl_dt = make_card(dash_cards, "dt")
    card_density, lbl_density = make_card(dash_cards, "Density")
    card_fps, lbl_fps = make_card(dash_cards, "FPS")

    card_particles.grid(row=0, column=0, sticky="ew", padx=5)
    card_dt.grid(row=0, column=1, sticky="ew", padx=5)
    card_density.grid(row=0, column=2, sticky="ew", padx=5)
    card_fps.grid(row=0, column=3, sticky="ew", padx=5)

    ttkmod.Label(page_dashboard, text="StarSimX is an educational N-body simulator for exploring star formation via gravitational collapse.")

    ttkmod.Label(page_dashboard, text="To start a simulation:").pack(anchor="w", padx=10, pady=(12, 0))
    ttkmod.Label(page_dashboard, text="- Adjust the parameters in the 'Parameters' panel on the right.").pack(anchor="w", padx=10)
    ttkmod.Label(page_dashboard, text="- Click 'Start' to run the simulation.").pack(anchor="w", padx=10)
    ttkmod.Label(page_dashboard, text="- Use 'Stop' to halt the simulation.").pack(anchor="w", padx=10)

    ttkmod.Label(page_dashboard, text="To view analytics:").pack(anchor="w", padx=10, pady=(12, 0))
    ttkmod.Label(page_dashboard, text="- Click on the 'Analytics' tab after a simulation run.").pack(anchor="w", padx=10)
    ttkmod.Label(page_dashboard, text="- Select a run to view energy drift and density proxy over time.").pack(anchor="w", padx=10)

    ttkmod.Label(page_dashboard, text="For best results:").pack(anchor="w", padx=10, pady=(12, 0))
    ttkmod.Label(page_dashboard, text="- Use N <= 2000 for faster computations.").pack(anchor="w", padx=10)
    ttkmod.Label(page_dashboard, text="- Keep an eye on the energy drift (% change) in the Analytics tab.").pack(anchor="w", padx=10)

    # Remove the short hint and add a full in-app instruction block
    # ttkmod.Label(page_dashboard, text="Welcome to StarSimX. Use the Simulation section to run a scenario.",
    #              foreground=muted).pack(anchor="w", padx=4)

    instructions_str = (
        "Welcome to StarSimX - an educational N-body simulator for exploring gravitational collapse.\n\n"
        "Quick guide:\n"
        "1) Simulation\n"
        "   - Set Parameters on the right: N (particles), radius, eps (softening), dt (time step), t_end,\n"
        "     vrms (initial velocity dispersion), omega (solid-body rotation), seed, init_mode.\n"
        "   - Optional: Save GIF to write an on-the-fly animation of the x-y collapse.\n"
        "   - Click Start to run. You can Stop to cancel early.\n"
        "   - The left plot shows particle positions; the panel shows live t, step, FPS, and a density proxy.\n"
        "   - UI update stride controls how often the plot updates (higher = smoother simulation, fewer UI refreshes).\n"
        "   - Diag stride controls how often energies/density are sampled for Analytics.\n"
        "   - Point size changes marker size in the plot.\n\n"
        "2) Analytics\n"
        "   - Open the Analytics page to review results after a run.\n"
        "   - Pick a run from the dropdown (based on out_prefix).\n"
        "   - Energies (K, U, Total) and Density proxy are plotted vs time.\n"
        "   - Smooth density applies a short moving average to the density curve.\n"
        "   - Export CSV saves t, K, U, E, D to a CSV file for external analysis.\n\n"
        "3) Outputs\n"
        "   - <out_prefix>_summary.png: final 3D scatter + energy curves.\n"
        "   - <out_prefix>_density.png: density proxy over time.\n"
        "   - <out_prefix>_data.npz: time series used by the Analytics page.\n"
        "   - Optional GIF if enabled.\n\n"
        "Tips\n"
        "   - Dark toggle in the top bar switches theme (and styles plots).\n"
        "   - Lower N or increase eps if the simulation is slow or noisy.\n"
        "   - Use a fixed seed to reproduce runs.\n"
        "   - CLI mode is also available: run StarSimX.py --cli (see README for examples).\n"
    )

    instr = ttkmod.Labelframe(page_dashboard, text="How to use the app", padding=(10, 8))
    if use_bootstrap:
        instr.configure(bootstyle="secondary")
    instr.pack(fill=tk.BOTH, expand=False, padx=6, pady=(6, 0), anchor="nw")

    try:
        from tkinter import scrolledtext as _st
        stw = _st.ScrolledText(instr, wrap="word", height=16)
        try:
            stw.configure(font=FONT_SM)
        except Exception:
            pass
        stw.insert("1.0", instructions_str)
        stw.configure(state="disabled")
        stw.pack(fill=tk.BOTH, expand=True)
    except Exception:
        ttkmod.Label(instr, text=instructions_str, justify="left", foreground=muted, wraplength=980).pack(anchor="w")

    # Simulation page
    page_sim = ttkmod.Frame(content)
    page_sim.grid(row=0, column=0, sticky="nsew")
    pages["Simulation"] = page_sim

    # Split: left plot fixed size, right params
    sim_left = ttkmod.Labelframe(page_sim, text="Simulation View", padding=8)
    sim_right = ttkmod.Labelframe(page_sim, text="Parameters", padding=10)
    sim_left.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 0))
    sim_right.grid(row=0, column=1, sticky="ns")
    page_sim.grid_columnconfigure(0, weight=1)
    page_sim.grid_rowconfigure(0, weight=1)

    # Fix geometry to avoid shaking
    CANVAS_W, CANVAS_H = 760, 560
    sim_left.configure(width=CANVAS_W + 20, height=CANVAS_H + 60)
    try:
        sim_left.grid_propagate(False)
    except Exception:
        pass

    # Matplotlib figure (constant title; use overlay text for time)
    fig = Figure(figsize=(CANVAS_W / 100, CANVAS_H / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Collapse (x-y)")
    ax.set_aspect("equal")
    point_size_var = tk.IntVar(value=2)
    scat = ax.scatter([], [], s=point_size_var.get())
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left', fontsize=9)

    canvas = FigureCanvasTkAgg(fig, master=sim_left)
    toolbar = NavigationToolbar2Tk(canvas, sim_left, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)
    canvas.get_tk_widget().configure(width=CANVAS_W, height=CANVAS_H)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Right parameters panel content
    params = {
        "N": tk.StringVar(value="600"),
        "radius": tk.StringVar(value="1.0"),
        "eps": tk.StringVar(value="0.02"),
        "dt": tk.StringVar(value="0.002"),
        "t_end": tk.StringVar(value="5.0"),
        "vrms": tk.StringVar(value="0.1"),
        "omega": tk.StringVar(value="0.0"),
        "seed": tk.StringVar(value="42"),
        "init_mode": tk.StringVar(value="uniform_sphere"),
        "gif_path": tk.StringVar(value=""),
        "out_prefix": tk.StringVar(value="run"),
    }

    def add_row(label, var, row):
        ttkmod.Label(sim_right, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttkmod.Entry(sim_right, textvariable=var, width=12).grid(row=row, column=1, sticky="w", pady=2)

    add_row("N", params["N"], 0)
    add_row("radius", params["radius"], 1)
    add_row("eps", params["eps"], 2)
    add_row("dt", params["dt"], 3)
    add_row("t_end", params["t_end"], 4)
    add_row("vrms", params["vrms"], 5)
    add_row("omega", params["omega"], 6)
    add_row("seed", params["seed"], 7)

    ttkmod.Label(sim_right, text="init_mode").grid(row=8, column=0, sticky="w", pady=2)
    if use_bootstrap:
        cmb = ttkmod.Combobox(sim_right, textvariable=params["init_mode"], values=["uniform_sphere", "gaussian"], width=14, state="readonly")
    else:
        import tkinter.ttk as ttk
        cmb = ttk.Combobox(sim_right, textvariable=params["init_mode"], values=["uniform_sphere", "gaussian"], width=14, state="readonly")
    cmb.grid(row=8, column=1, sticky="w", pady=2)

    # GIF controls
    gif_chk_var = tk.BooleanVar(value=False)
    def browse_gif():
        path = filedialog.asksaveasfilename(title="Save animation GIF", defaultextension=".gif", filetypes=[("GIF","*.gif")])
        if path:
            params["gif_path"].set(path)
            gif_chk_var.set(True)
    ttkmod.Checkbutton(sim_right, text="Save GIF", variable=gif_chk_var).grid(row=9, column=0, sticky="w", pady=(8, 2))
    gif_row = ttkmod.Frame(sim_right)
    gif_row.grid(row=9, column=1, sticky="w", pady=(8, 2))
    ttkmod.Entry(gif_row, textvariable=params["gif_path"], width=14).pack(side=tk.LEFT)
    ttkmod.Button(gif_row, text="Browse", command=browse_gif).pack(side=tk.LEFT, padx=(4, 0))

    # out_prefix row
    add_row("out_prefix", params["out_prefix"], 10)

    # Actions and status
    actions = ttkmod.Frame(sim_right)
    actions.grid(row=11, column=0, columnspan=2, pady=(10, 6), sticky="ew")
    actions.grid_columnconfigure((0,1), weight=1)
    btn_start = ttkmod.Button(actions, text="Start", width=12)
    btn_stop = ttkmod.Button(actions, text="Stop", width=12)
    btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    btn_stop.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    btn_load = ttkmod.Button(sim_right, text="Load Config")
    btn_save = ttkmod.Button(sim_right, text="Save Config")
    btn_load.grid(row=12, column=0, sticky="ew", pady=2)
    btn_save.grid(row=12, column=1, sticky="ew", pady=2)

    progress = ttkmod.Progressbar(sim_right, mode="determinate")
    progress.grid(row=13, column=0, columnspan=2, pady=(10, 4), sticky="ew")

    status = tk.StringVar(value="Idle")
    ttkmod.Label(sim_right, textvariable=status).grid(row=14, column=0, columnspan=2, sticky="w")

    stats = tk.StringVar(value="t=0.00 | step=0 | fps=0.0 | density?- | N=0")
    ttkmod.Label(sim_right, textvariable=stats, foreground=muted, width=60, anchor="w").grid(row=15, column=0, columnspan=2, sticky="w")

    # Performance controls
    ttkmod.Label(sim_right, text="UI update stride").grid(row=16, column=0, sticky="w", pady=(6,0))
    ui_stride_var = tk.IntVar(value=1)
    ttkmod.Spinbox(sim_right, from_=1, to=50, textvariable=ui_stride_var, width=8).grid(row=16, column=1, sticky="w", pady=(6,0))

    ttkmod.Label(sim_right, text="Diag stride (energy/density)").grid(row=17, column=0, sticky="w")
    diag_stride_var = tk.IntVar(value=10)
    ttkmod.Spinbox(sim_right, from_=1, to=200, textvariable=diag_stride_var, width=8).grid(row=17, column=1, sticky="w")

    ttkmod.Label(sim_right, text="Point size").grid(row=18, column=0, sticky="w")
    ttkmod.Spinbox(sim_right, from_=1, to=10, textvariable=point_size_var, width=8).grid(row=18, column=1, sticky="w")

    # Analytics context/state (lazy init to avoid startup lag)
    analytics_ctx = {"inited": False}

    # Analytics page placeholder (frame only now)
    page_analytics = ttkmod.Frame(content)
    page_analytics.grid(row=0, column=0, sticky="nsew")
    pages["Analytics"] = page_analytics

    def init_analytics():
        if analytics_ctx["inited"]:
            return
        import glob, csv, threading as _threading, os as _os
        # Top bar within analytics
        ana_top = ttkmod.Frame(page_analytics)
        ana_top.pack(fill=tk.X, pady=6)
        ttkmod.Label(ana_top, text="Analytics", font=FONT_BOLD_MD).pack(side=tk.LEFT)

        run_var = tk.StringVar(value="")
        run_select = ttkmod.Combobox(ana_top, textvariable=run_var, width=28, state="readonly")
        run_select.pack(side=tk.LEFT, padx=(10, 6))

        smooth_var = tk.BooleanVar(value=False)
        ttkmod.Checkbutton(ana_top, text="Smooth density", variable=smooth_var).pack(side=tk.LEFT)

        btn_refresh = ttkmod.Button(ana_top, text="Refresh")
        btn_refresh.pack(side=tk.RIGHT)
        btn_export = ttkmod.Button(ana_top, text="Export CSV")
        btn_export.pack(side=tk.RIGHT, padx=(0, 6))

        # Loading indicator
        loading_var = tk.StringVar(value="")
        ttkmod.Label(page_analytics, textvariable=loading_var, foreground=muted).pack(anchor="w")

        # Metrics row
        metrics = ttkmod.Frame(page_analytics)
        metrics.pack(fill=tk.X)
        ttkmod.Label(metrics, text="Energy drift:").grid(row=0, column=0, sticky="w")
        lbl_drift = ttkmod.Label(metrics, text="-", foreground=muted)
        lbl_drift.grid(row=0, column=1, sticky="w", padx=(4, 12))

        ttkmod.Label(metrics, text="Collapse time (max density):").grid(row=0, column=2, sticky="w")
        lbl_tcol = ttkmod.Label(metrics, text="-", foreground=muted)
        lbl_tcol.grid(row=0, column=3, sticky="w", padx=(4, 12))

        ttkmod.Label(metrics, text="Samples:").grid(row=0, column=4, sticky="w")
        lbl_samples = ttkmod.Label(metrics, text="-", foreground=muted)
        lbl_samples.grid(row=0, column=5, sticky="w", padx=(4, 12))

        # Plots area
        ana_plots = ttkmod.Frame(page_analytics)
        ana_plots.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        ana_plots.columnconfigure(0, weight=1)
        ana_plots.columnconfigure(1, weight=1)
        ana_plots.rowconfigure(0, weight=1)

        # Energy figure
        fig_e = Figure(figsize=(5.5, 3.8), dpi=100)
        ax_e = fig_e.add_subplot(111)
        ax_e.set_title("Energies vs time")
        ax_e.set_xlabel("t")
        ax_e.set_ylabel("Energy (arb.)")
        line_K, = ax_e.plot([], [], label="Kinetic", color="#3b82f6")
        line_U, = ax_e.plot([], [], label="Potential", color="#ef4444")
        line_E, = ax_e.plot([], [], label="Total", color="#10b981")
        ax_e.legend(loc="best")
        canvas_e = FigureCanvasTkAgg(fig_e, master=ana_plots)
        canvas_e.draw()
        canvas_e.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6)

        # Density figure
        fig_d = Figure(figsize=(5.5, 3.8), dpi=100)
        ax_d = fig_d.add_subplot(111)
        ax_d.set_title("Density proxy vs time")
        ax_d.set_xlabel("t")
        ax_d.set_ylabel("1 / <d_kNN>")
        line_D, = ax_d.plot([], [], label="Density proxy", color="#8b5cf6")
        mark_tcol = ax_d.axvline(0, color="#6b7280", linestyle="--", alpha=0.6)
        canvas_d = FigureCanvasTkAgg(fig_d, master=ana_plots)
        canvas_d.draw()
        canvas_d.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=6)

        # Store in context
        analytics_ctx.update({
            "inited": True,
            "run_var": run_var,
            "run_select": run_select,
            "smooth_var": smooth_var,
            "btn_refresh": btn_refresh,
            "btn_export": btn_export,
            "loading_var": loading_var,
            "lbl_drift": lbl_drift,
            "lbl_tcol": lbl_tcol,
            "lbl_samples": lbl_samples,
            "ax_e": ax_e, "ax_d": ax_d,
            "canvas_e": canvas_e, "canvas_d": canvas_d,
            "line_K": line_K, "line_U": line_U, "line_E": line_E,
            "line_D": line_D, "mark_tcol": mark_tcol,
            "load_id": 0,
        })

        # Dark styling if toggle is on
        if dark_var.get():
            toggle_dark()

        def list_runs():
            files = glob.glob("*_data.npz")
            files.sort(key=lambda p: _os.path.getmtime(p), reverse=True)
            return [ _os.path.basename(p).replace("_data.npz", "") for p in files ]

        def moving_average(x, w=7):
            if w <= 1 or len(x) < w:
                return x
            import numpy as _np
            c = _np.convolve(x, _np.ones(w)/w, mode='valid')
            pad = [x[0]] * (w-1)
            return _np.concatenate([pad, c])

        def set_analytics_data(t, K, U, D, token):
            # Apply only if this is the latest load
            if token != analytics_ctx.get("load_id"):
                return
            ax_e = analytics_ctx["ax_e"]; ax_d = analytics_ctx["ax_d"]
            line_K = analytics_ctx["line_K"]; line_U = analytics_ctx["line_U"]; line_E = analytics_ctx["line_E"]
            line_D = analytics_ctx["line_D"]; mark_tcol = analytics_ctx["mark_tcol"]
            canvas_e = analytics_ctx["canvas_e"]; canvas_d = analytics_ctx["canvas_d"]

            E = K + U
            line_K.set_data(t, K)
            line_U.set_data(t, U)
            line_E.set_data(t, E)
            ax_e.relim(); ax_e.autoscale()

            D_plot = moving_average(D) if analytics_ctx["smooth_var"].get() else D
            line_D.set_data(t, D_plot)
            ax_d.relim(); ax_d.autoscale()

            drift = 0.0
            if len(E) > 1 and abs(E[0]) > 1e-12:
                drift = (E[-1] - E[0]) / abs(E[0]) * 100.0
            analytics_ctx["lbl_drift"].configure(text=f"{drift:+.2f}%")
            if len(D) > 0:
                i_max = int(np.argmax(D))
                tcol = float(t[i_max])
                analytics_ctx["lbl_tcol"].configure(text=f"{tcol:.3f}")
                # axvline returns a Line2D that expects sequences for set_xdata
                # pass [tcol, tcol] to define the vertical line's x at both ends
                mark_tcol.set_xdata([tcol, tcol])
            analytics_ctx["lbl_samples"].configure(text=str(len(t)))

            canvas_e.draw_idle(); canvas_d.draw_idle()
            analytics_ctx["loading_var"].set("")

        def update_analytics_async(prefix: str):
            if not prefix:
                return
            analytics_ctx["load_id"] += 1
            token = analytics_ctx["load_id"]
            analytics_ctx["loading_var"].set(f"Loading: {prefix}_data.npz ...")
            def worker():
                try:
                    path = f"{prefix}_data.npz"
                    if not _os.path.exists(path):
                        root.after(0, lambda: analytics_ctx["loading_var"].set(f"Not found: {path}"))
                        return
                    data = np.load(path)
                    t = data["t"].astype(float)
                    K = data["K"].astype(float)
                    U = data["U"].astype(float)
                    D = data["D"].astype(float)
                    root.after(0, lambda: set_analytics_data(t, K, U, D, token))
                except Exception as e:
                    root.after(0, lambda: analytics_ctx["loading_var"].set(f"Error: {e}"))
            _threading.Thread(target=worker, daemon=True).start()

        def refresh_runs(select_latest=True):
            prefixes = list_runs()
            analytics_ctx["run_select"].configure(values=prefixes)
            if prefixes:
                if select_latest or analytics_ctx["run_var"].get() not in prefixes:
                    analytics_ctx["run_var"].set(prefixes[0])
                update_analytics_async(analytics_ctx["run_var"].get())
            else:
                analytics_ctx["run_var"].set("")
                analytics_ctx["loading_var"].set("No runs found. Run a simulation first.")

        def export_csv():
            pfx = analytics_ctx["run_var"].get()
            if not pfx:
                messagebox.showinfo("Export", "No run selected.")
                return
            try:
                data = np.load(f"{pfx}_data.npz")
                t = data["t"]; K = data["K"]; U = data["U"]; D = data["D"]; E = K + U
                out = f"{pfx}_timeseries.csv"
                with open(out, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "K", "U", "E", "D"])
                    for i in range(len(t)):
                        w.writerow([float(t[i]), float(K[i]), float(U[i]), float(E[i]), float(D[i])])
                messagebox.showinfo("Export", f"Saved: {out}")
            except Exception as e:
                messagebox.showerror("Export", f"Failed to export CSV:\n{e}")

        # Bindings/buttons
        analytics_ctx["btn_refresh"].configure(command=lambda: refresh_runs(select_latest=False))
        analytics_ctx["btn_export"].configure(command=export_csv)
        run_select.bind("<<ComboboxSelected>>", lambda _e: update_analytics_async(analytics_ctx["run_var"].get()))

        # First population
        refresh_runs(select_latest=True)

        # Store helpers for external calls
        analytics_ctx["refresh_runs"] = refresh_runs
        analytics_ctx["update_async"] = update_analytics_async

    # Update dark mode to style analytics axes if they exist
    def safe_style_analytics_on_theme_toggle():
        if analytics_ctx.get("inited"):
            try:
                # trigger re-style via toggle_dark (it already styles ax_e/ax_d if present)
                pass
            except Exception:
                pass

    # Replace toggle_dark callsite hook left as-is; it checks and styles axes safely

    # Page switching logic
    active_page = {"name": "Dashboard"}

    def show_page(name: str):
        # Hide all
        for f in pages.values():
            f.grid_remove()
        # Lazily init analytics
        if name == "Analytics":
            init_analytics()
            if analytics_ctx.get("refresh_runs"):
                try:
                    analytics_ctx["refresh_runs"](select_latest=False)
                except Exception:
                    pass
        # Show target
        pages[name].grid()
        active_page["name"] = name
        title.configure(text=name)

    # Sidebar buttons after pages so callbacks can reference show_page
    nav = [
        ("Dashboard", lambda: show_page("Dashboard")),
        ("Simulation", lambda: show_page("Simulation")),
        ("Analytics", lambda: show_page("Analytics")),
    ]
    for i, (text, cmd) in enumerate(nav, start=1):
        ttkmod.Button(sidebar, text=text, command=cmd, width=18).grid(row=i, column=0, sticky="ew", pady=2)

    # Default page
    show_page("Dashboard")

    # cancel and state
    cancel_event = threading.Event()
    running = {"active": False}
    try:
        btn_stop.state(["disabled"])  # type: ignore[attr-defined]
    except Exception:
        pass

    # Config helpers
    def read_cfg_from_ui():
        return {
            "N": int(params["N"].get()),
            "radius": float(params["radius"].get()),
            "eps": float(params["eps"].get()),
            "dt": float(params["dt"].get()),
            "t_end": float(params["t_end"].get()),
            "vrms": float(params["vrms"].get()),
            "omega": float(params["omega"].get()),
            "seed": int(params["seed"].get()) if params["seed"].get() != "" else None,
            "init_mode": params["init_mode"].get(),
            "gif": params["gif_path"].get() if gif_chk_var.get() and params["gif_path"].get() else None,
            "out_prefix": params["out_prefix"].get() or "run",
        }

    def write_cfg_to_ui(cfg: dict):
        for k, v in cfg.items():
            if k in params:
                params[k].set(str(v))
        if cfg.get("gif"):
            params["gif_path"].set(cfg["gif"])
            gif_chk_var.set(True)

    def load_cfg():
        path = filedialog.askopenfilename(title="Load config JSON", filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            write_cfg_to_ui(cfg)
            status.set("Loaded config")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    def save_cfg():
        path = filedialog.asksaveasfilename(title="Save config JSON", defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            cfg = read_cfg_from_ui()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            status.set("Saved config")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{e}")

    btn_load.configure(command=load_cfg)
    btn_save.configure(command=save_cfg)

    # Start/Stop behavior
    def start():
        if running["active"]:
            return
        try:
            cfg = read_cfg_from_ui()
        except Exception:
            status.set("Invalid parameter value")
            return

        cancel_event.clear()
        running["active"] = True
        try:
            btn_start.state(["disabled"])  # type: ignore[attr-defined]
            btn_stop.state(["!disabled"])  # type: ignore[attr-defined]
        except Exception:
            pass
        status.set("Running...")
        nsteps = max(1, int(cfg["t_end"] / cfg["dt"]))
        progress.configure(mode="determinate")
        progress["maximum"] = nsteps
        progress["value"] = 0

        # Go to Simulation page when starting
        show_page("Simulation")

        # Reset plot
        scat.set_offsets(np.empty((0, 2)))
        try:
            scat.set_sizes([float(point_size_var.get())])
        except Exception:
            pass
        ax.relim(); ax.autoscale(False)
        lim = cfg["radius"] * 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        time_text.set_text("")
        canvas.draw()

        timings = {"t0": time.perf_counter(), "last_ts": time.perf_counter(), "last_step": 0}
        density_text = {"val": None}

        # KPI static
        lbl_particles.configure(text=str(cfg["N"]))
        lbl_dt.configure(text=str(cfg["dt"]))

        ui_stride = max(1, int(ui_stride_var.get() or 1))
        diag_stride = max(1, int(diag_stride_var.get() or 10))

        def ui_update(t, step, pos, vel):
            def _apply():
                if not running["active"] or cancel_event.is_set():
                    return
                xy = pos[:, :2]
                scat.set_offsets(xy)
                # update fps/density occasionally
                now = time.perf_counter()
                dt_wall = now - timings["last_ts"]
                fps = (step - timings["last_step"]) / dt_wall if dt_wall > 1e-6 else 0.0
                timings["last_ts"] = now
                timings["last_step"] = step
                if step % 20 == 0:
                    try:
                        density_text["val"] = f"{nn_density_proxy(pos, k=8):.3f}"
                    except Exception:
                        density_text["val"] = "-"
                progress["value"] = step
                time_text.set_text(f"t={t:.2f}  step={step}")
                stats.set(
                    f"t={t:.2f} | step={step} | fps={fps:.1f} | density?{density_text['val'] or '-'} | N={xy.shape[0]}"
                )
                lbl_fps.configure(text=f"{fps:.1f}")
                if density_text["val"]:
                    lbl_density.configure(text=str(density_text["val"]))
                canvas.draw_idle()
            root.after(0, _apply)

        def worker():
            try:
                simulate(
                    N=cfg["N"], radius=cfg["radius"], eps=cfg["eps"], dt=cfg["dt"], t_end=cfg["t_end"],
                    vrms=cfg["vrms"], omega=cfg["omega"], seed=cfg["seed"], init_mode=cfg["init_mode"],
                    out_prefix=cfg["out_prefix"], step_callback=ui_update, callback_stride=ui_stride, stop_event=cancel_event,
                    save_gif=cfg.get("gif"), energy_stride=diag_stride, headless=False
                )
            finally:
                def _done():
                    status.set("Idle")
                    running["active"] = False
                    try:
                        btn_start.state(["!disabled"])  # type: ignore[attr-defined]
                        btn_stop.state(["disabled"])  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    # Refresh analytics list if initialized
                    if analytics_ctx.get("inited") and analytics_ctx.get("refresh_runs"):
                        try:
                            analytics_ctx["refresh_runs"](select_latest=False)
                        except Exception:
                            pass
                root.after(0, _done)

        threading.Thread(target=worker, daemon=True).start()

    def stop():
        if running["active"]:
            cancel_event.set()
            status.set("Stopping...")

    btn_start.configure(command=start)
    btn_stop.configure(command=stop)

    root.mainloop()


# --- CLI glue ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Educational N-body collapse simulator")
    p.add_argument("--N", type=int, default=800)
    p.add_argument("--mass_total", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--t_end", type=float, default=5.0)
    p.add_argument("--vrms", type=float, default=0.1)
    p.add_argument("--omega", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--init_mode", choices=["uniform_sphere", "gaussian"], default="uniform_sphere")
    p.add_argument("--gif", type=str, default=None, help="path to save GIF animation")
    p.add_argument("--gif_stride", type=int, default=5)
    p.add_argument("--energy_stride", type=int, default=10)
    p.add_argument("--out_prefix", type=str, default="run")
    p.add_argument("--config", type=str, default=None, help="JSON config file (overrides CLI defaults)")
    # UI selection
    p.add_argument("--gui", action="store_true", help="launch GUI app")
    p.add_argument("--cli", action="store_true", help="run in CLI (no window)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    try:
        # Prefer GUI unless --cli is explicitly requested
        if len(sys.argv) == 1:
            launch_gui()
            return

        args = parse_args()

        if getattr(args, "gui", False) or not getattr(args, "cli", False):
            # If --gui, or no explicit --cli, launch GUI
            launch_gui()
            return

        # CLI path
        cfg = vars(args)
        if args.config:
            cfg_json = load_config(args.config)
            cfg.update(cfg_json)

        simulate(
            N=cfg["N"],
            mass_total=cfg.get("mass_total", 1.0),
            radius=cfg["radius"],
            eps=cfg["eps"],
            dt=cfg["dt"],
            t_end=cfg["t_end"],
            vrms=cfg["vrms"],
            omega=cfg["omega"],
            seed=cfg["seed"],
            init_mode=cfg["init_mode"],
            save_gif=cfg.get("gif"),
            gif_stride=cfg.get("gif_stride", 5),
            out_prefix=cfg.get("out_prefix", "run"),
            energy_stride=cfg.get("energy_stride", 10),
            headless=True,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        if sys.gettrace() is not None:
            try:
                input("An error occurred. Press Enter to exit...")
            except Exception:
                pass

if __name__ == "__main__":
    main()
