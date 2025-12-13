import sys
import numpy as np
from pathlib import Path
import matplotlib

# choose matplotlib backend and set _USE_TK flag
_USE_TK = False
try:
    # If running inside IPython/Jupyter, prefer the widget backend
    get_ipython().run_line_magic('matplotlib', 'widget')
    _USE_TK = False
except Exception:
    # Not in notebook or magic unavailable — try a GUI backend, fall back to Agg
    try:
        matplotlib.use('TkAgg')
        _USE_TK = True
    except Exception:
        matplotlib.use('Agg')
        _USE_TK = False

# import pyplot after backend selection
import matplotlib.pyplot as plt

from ikpy.chain import Chain
from ikpy.utils import plot
from ikpy.link import OriginLink, URDFLink

_here = Path(__file__).resolve().parent

# Use the current folder (SNC_R4) as the resources directory and load all .json chain files
_res_dir = _here

json_files = sorted(_res_dir.glob("*.json"))
chains = {}

if not json_files:
    print("No .json chain files found in resources folder:", _res_dir)
    print("Creating simple demo chains for simulation (left/right arms + legs).")

    def make_arm(name, side='left'):
        return Chain(name=name, links=[
            OriginLink(),
            URDFLink(name=f"{side}_shoulder_pitch", translation_vector=[0, 0, 0.05],
                     orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=[-np.pi/2, np.pi/2]),
            URDFLink(name=f"{side}_shoulder_roll", translation_vector=[0, 0, 0.02],
                     orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=[-np.pi/2, np.pi/2]),
            URDFLink(name=f"{side}_elbow", translation_vector=[0, 0, 0.15],
                     orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=[-np.pi, np.pi]),
        ])

    def make_leg(name, side='left'):
        return Chain(name=name, links=[
            OriginLink(),
            URDFLink(name=f"{side}_hip_pitch", translation_vector=[0, 0, 0.05],
                     orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=[-np.pi/2, np.pi/2]),
            URDFLink(name=f"{side}_knee", translation_vector=[0, 0, 0.18],
                     orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=[-np.pi, 0]),
            URDFLink(name=f"{side}_ankle", translation_vector=[0, 0, 0.10],
                     orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=[-np.pi/2, np.pi/2]),
        ])

    chains = {
        "left_arm_demo": make_arm("left_arm_demo", "left"),
        "right_arm_demo": make_arm("right_arm_demo", "right"),
        "left_leg_demo": make_leg("left_leg_demo", "left"),
        "right_leg_demo": make_leg("right_leg_demo", "right"),
    }
else:
    for jf in json_files:
        try:
            chains[jf.stem] = Chain.from_json_file(str(jf))
        except Exception as e:
            print(f"Failed to load {jf.name}: {e}")

print("Using resources from:", _res_dir)
print("Loaded chain files:", ", ".join(chains.keys()))

# DEBUG / auto-fix: print chain info and ensure active_links_mask is set
for name, ch in chains.items():
    try:
        link_names = [ln.name for ln in ch.links]
    except Exception:
        link_names = None

    mask = getattr(ch, "active_links_mask", None)

    # safe counts for mask (handle numpy arrays)
    mask_len = len(mask) if mask is not None else "n/a"
    try:
        active_count = int(np.count_nonzero(mask)) if mask is not None else 0
    except Exception:
        try:
            active_count = sum(1 for m in mask if bool(m))
        except Exception:
            active_count = "n/a"

    print(f"Chain '{name}': links={len(ch.links)} mask_len={mask_len} active_count={active_count}")
    print("  link names:", link_names)
    print("  active_links_mask:", mask)

    # determine if mask is missing or all False (robust for numpy / lists)
    mask_missing_or_all_false = False
    if mask is None:
        mask_missing_or_all_false = True
    else:
        try:
            if mask_len == 0 or not bool(np.any(mask)):
                mask_missing_or_all_false = True
        except Exception:
            try:
                if all((not bool(m)) for m in mask):
                    mask_missing_or_all_false = True
            except Exception:
                mask_missing_or_all_false = False

    # if mask is missing or all False, set a sensible default (enable inner joints)
    if mask_missing_or_all_false:
        default_mask = [False] + [True] * (len(ch.links) - 2) + [False]
        try:
            ch.active_links_mask = default_mask
            print(f"  -> fixed active_links_mask for '{name}' to default ({len(default_mask)} entries)")
        except Exception as e:
            print(f"  -> failed to set active_links_mask for '{name}': {e}")

# helper to pick a chain by keywords (falls back to the first loaded chain)
def _pick_chain(*keywords):
    for name, ch in chains.items():
        ln = name.lower()
        if all(k.lower() in ln for k in keywords):
            return ch
    # fallback to first chain if no match found
    return next(iter(chains.values()))

# After chains loaded and initial debug print loop, add robustness helpers
def _ensure_active_mask(ch):
    """Ensure ch.active_links_mask exists and has correct length."""
    try:
        n = len(ch.links)
    except Exception:
        return
    mask = getattr(ch, "active_links_mask", None)
    # if missing or all False or wrong length -> set sensible default
    need_fix = False
    if mask is None:
        need_fix = True
    else:
        try:
            if len(mask) != n:
                need_fix = True
            else:
                # treat numpy array and lists: check if all False
                if not bool(np.any(mask)):
                    need_fix = True
        except Exception:
            need_fix = True
    if need_fix:
        default_mask = [False] + [True] * (n - 2) + [False] if n >= 2 else [False] * n
        try:
            ch.active_links_mask = default_mask
            print(f"  -> fixed active_links_mask for '{ch.name}' to default ({len(default_mask)} entries)")
        except Exception as e:
            print(f"  -> failed to set active_links_mask for '{ch.name}': {e}")

def _chain_has_terms(ch, terms):
    """Return True if all terms appear in chain name or in any link name."""
    ln = ch.name.lower() if hasattr(ch, "name") else ""
    if all(t in ln for t in terms):
        return True
    try:
        link_names = [getattr(l, "name", "").lower() for l in ch.links]
        for t in terms:
            if not any(t in ln2 for ln2 in link_names):
                return False
        return True
    except Exception:
        return False

# Ensure masks are valid for all chains (reinforce earlier fixes)
for name, ch in chains.items():
    _ensure_active_mask(ch)

# Replace _pick_chain with a more robust version that also checks link names
def _pick_chain(*keywords):
    # first try existing filename-based matching
    for name, ch in chains.items():
        ln = name.lower()
        if all(k.lower() in ln for k in keywords):
            return ch
    # then try matching by chain.name
    for name, ch in chains.items():
        try:
            if all(k.lower() in getattr(ch, "name", "").lower() for k in keywords):
                return ch
        except Exception:
            pass
    # then try matching by link names (useful for generated chains)
    for name, ch in chains.items():
        if _chain_has_terms(ch, [k.lower() for k in keywords]):
            return ch
    # fallback: prefer chain that looks like an arm (has shoulder/elbow) for keywords containing "arm"
    if "arm" in " ".join(keywords).lower():
        for name, ch in chains.items():
            try:
                lnks = [getattr(l, "name", "").lower() for l in ch.links]
                if any("shoulder" in s for s in lnks) or any("elbow" in s for s in lnks):
                    return ch
            except Exception:
                pass
    # final fallback to first chain
    return next(iter(chains.values()))

# Helper that calls inverse_kinematics robustly and returns full-length vector (len == len(chain.links))
def safe_inverse_kinematics(chain, target, initial_position=None):
    t = np.asarray(target)
    trys = []
    # try shapes: (3,), (4,) with homogeneous 1
    if t.size == 3:
        trys.append(t)
        trys.append(np.append(t, 1.0))
    elif t.size == 4:
        trys.append(t[:3])
        trys.append(t)
    else:
        trys.append(t.flatten()[:3])
        trys.append(np.append(t.flatten()[:3], 1.0))
    last_exc = None
    for tt in trys:
        try:
            res = chain.inverse_kinematics(tt, initial_position=initial_position)
            # ensure result is numpy array
            res = np.asarray(res)
            # pad or truncate to match number of links
            n = len(chain.links)
            if res.size < n:
                res = np.concatenate([res, np.zeros(n - res.size)])
            elif res.size > n:
                res = res[:n]
            return res
        except Exception as e:
            last_exc = e
            continue
    print(f"safe_inverse_kinematics failed for chain '{getattr(chain, 'name', '')}': {last_exc}")
    # final fallback: return zeros
    return np.zeros(len(chain.links))

# try to pick the common robot parts by name; adjust keywords if your files use different naming
robotis_op3_left_arm_chain  = _pick_chain("left", "arm")
robotis_op3_right_arm_chain = _pick_chain("right", "arm")
robotis_op3_left_leg_chain  = _pick_chain("left", "leg")
robotis_op3_right_leg_chain = _pick_chain("right", "leg")

print("Selected chains:",
      robotis_op3_left_arm_chain.name,
      robotis_op3_right_arm_chain.name,
      robotis_op3_left_leg_chain.name,
      robotis_op3_right_leg_chain.name)

# ----- Realtime UI + simulation -----
# If running in a GUI-capable environment embed matplotlib in a Tk window
if _USE_TK:
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("IKPy Realtime Simulator - Robotis OP3 (both arms)")

    # --- Scrollable Control Frame Setup ---
    # Create a main container frame for the controls that will hold the canvas and scrollbar
    main_ctrl_frame = tk.Frame(root)
    main_ctrl_frame.pack(side=tk.LEFT, fill="y", padx=8, pady=8)

    # Create a canvas and a scrollbar
    canvas_ctrl = tk.Canvas(main_ctrl_frame)
    scrollbar = ttk.Scrollbar(main_ctrl_frame, orient="vertical", command=canvas_ctrl.yview)
    canvas_ctrl.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas_ctrl.pack(side="left", fill="both", expand=True)

    # Create a frame inside the canvas to hold the controls
    ctrl = tk.Frame(canvas_ctrl)
    canvas_ctrl.create_window((0, 0), window=ctrl, anchor="nw")

    # Update scrollregion when the inner frame's size changes
    def on_frame_configure(event):
        canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all"))
    ctrl.bind("<Configure>", on_frame_configure)

    # --- Mode Selection ---
    mode_frame = tk.LabelFrame(ctrl, text="Mode", padx=6, pady=6)
    mode_frame.pack(padx=4, pady=4, fill=tk.X)
    control_mode = tk.StringVar(value="FK")

    # --- View Controls ---
    def set_view(elev, azim):
        ax.view_init(elev=elev, azim=azim)
        canvas.draw_idle()

    view_frame = tk.LabelFrame(ctrl, text="View Control", padx=6, pady=6)
    view_frame.pack(padx=4, pady=4, fill=tk.X)
    
    view_button_frame = tk.Frame(view_frame)
    view_button_frame.pack()

    tk.Button(view_button_frame, text="Front", command=lambda: set_view(elev=20, azim=-90)).pack(side=tk.LEFT, padx=2)
    tk.Button(view_button_frame, text="Side", command=lambda: set_view(elev=20, azim=0)).pack(side=tk.LEFT, padx=2)
    tk.Button(view_button_frame, text="Top", command=lambda: set_view(elev=90, azim=-90)).pack(side=tk.LEFT, padx=2)

    # Default targets for left and right arms
    target_left = {"x": 0.1, "y": 0.2, "z": 0.1}
    target_right = {"x": 0.1, "y": -0.2, "z": 0.1}
    running = {"val": True}
    update_interval_ms = 100  # update every 100 ms

    # helper to create labelled slider
    def make_slider(frame, label_text, initial, minv, maxv, step=0.01):
        lbl = tk.Label(frame, text=label_text)
        lbl.pack()
        var = tk.DoubleVar(value=initial)
        s = tk.Scale(frame, variable=var, from_=minv, to=maxv, resolution=step, orient=tk.HORIZONTAL, length=100)
        s.pack()
        return var

    # --- IK Controls ---
    ik_frame = tk.LabelFrame(ctrl, text="IK Controls", padx=2, pady=6)
    ik_frame.pack(padx=4, pady=4, fill=tk.X)

    # Create a container frame to hold left and right IK frames side-by-side
    ik_main_frame = tk.Frame(ik_frame)
    ik_main_frame.pack(fill=tk.X)


    # Left arm controls group
    left_frame = tk.LabelFrame(ik_main_frame, text="Left arm target", padx=6, pady=6)
    left_frame.pack(side=tk.LEFT, padx=2, pady=2, expand=True, fill=tk.X)
    lx = make_slider(left_frame, "x", target_left["x"], -0.5, 0.5)
    ly = make_slider(left_frame, "y", target_left["y"], -0.5, 0.5)
    lz = make_slider(left_frame, "z", target_left["z"], -0.8, 0.8)

    # Right arm controls group
    right_frame = tk.LabelFrame(ik_main_frame, text="Right arm target", padx=6, pady=6)
    right_frame.pack(side=tk.RIGHT, padx=2, pady=2, expand=True, fill=tk.X)
    rx = make_slider(right_frame, "x", target_right["x"], -0.5, 0.5)
    ry = make_slider(right_frame, "y", target_right["y"], -0.5, 0.5)
    rz = make_slider(right_frame, "z", target_right["z"], -0.8, 0.8)

    # --- FK Controls ---
    fk_frame = tk.LabelFrame(ctrl, text="FK Controls", padx=6, pady=6)
    fk_frame.pack(padx=4, pady=4, fill=tk.X)
    
    # Create a container frame to hold left and right FK frames side-by-side
    fk_main_frame = tk.Frame(fk_frame)
    fk_main_frame.pack(fill=tk.X)

    left_fk_frame = tk.LabelFrame(fk_main_frame, text="Left Arm Joints", padx=6, pady=6)
    left_fk_frame.pack(side=tk.LEFT, padx=2, pady=2, expand=True, fill=tk.X)
    right_fk_frame = tk.LabelFrame(fk_main_frame, text="Right Arm Joints", padx=6, pady=6)
    right_fk_frame.pack(side=tk.RIGHT, padx=2, pady=2, expand=True, fill=tk.X)

    left_joint_vars = []
    right_joint_vars = []

    def setup_joint_sliders(frame, chain, joint_vars_list):
        joint_vars_list.clear()
        for widget in frame.winfo_children():
            widget.destroy()
        
        active_links_indices = [i for i, active in enumerate(chain.active_links_mask) if active]
        for i in active_links_indices:
            link = chain.links[i]
            # Use link name, fallback to a generic name
            link_name = getattr(link, 'name', f'Joint {i}')
            # Use URDF bounds if available, otherwise default
            min_b, max_b = getattr(link, 'bounds', (-np.pi, np.pi))
            # Initial value is 0
            var = make_slider(frame, link_name, 0, np.rad2deg(min_b), np.rad2deg(max_b), step=1)
            joint_vars_list.append(var)

    setup_joint_sliders(left_fk_frame, robotis_op3_left_arm_chain, left_joint_vars)
    setup_joint_sliders(right_fk_frame, robotis_op3_right_arm_chain, right_joint_vars)

    def set_ui_mode(*args):
        mode = control_mode.get()
        if mode == "IK":
            for child in ik_frame.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, tk.Scale): widget.config(state=tk.NORMAL)
            for child in fk_frame.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, tk.Scale): widget.config(state=tk.DISABLED)
        elif mode == "FK":
            for child in ik_frame.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, tk.Scale): widget.config(state=tk.DISABLED)
            for child in fk_frame.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, tk.Scale): widget.config(state=tk.NORMAL)

    tk.Radiobutton(mode_frame, text="IK (Inverse Kinematics)", variable=control_mode, value="IK", command=set_ui_mode).pack(anchor=tk.W)
    tk.Radiobutton(mode_frame, text="FK (Forward Kinematics)", variable=control_mode, value="FK", command=set_ui_mode).pack(anchor=tk.W)
    control_mode.trace("w", set_ui_mode)

    def toggle_running():
        running["val"] = not running["val"]
        btn.config(text="Stop" if running["val"] else "Start")

    info_label = tk.Label(ctrl, text="IK angles update live for both arms")
    info_label.pack(pady=4)

    # display current joint angles for both arms
    angles_text = tk.Text(ctrl, width=30, height=8)
    angles_text.pack(pady=4)

    btn = tk.Button(ctrl, text="Stop", command=toggle_running)
    btn.pack(pady=8)

    set_ui_mode() # Initial setup

    # Figure for plotting
    fig = plt.Figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.5,0.5); ax.set_ylim(-0.5,0.5); ax.set_zlim(0,0.8)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    # helper: ensure angles vector is full-length for Chain.plot
    def _ensure_full_angles(chain, angles):
        """Return a numpy array of length len(chain.links).
        Accepts:
         - full-length arrays -> returned untouched
         - active-only arrays -> converted via chain.full_from_active if available
         - other sizes -> padded/truncated to fit
        """
        arr = np.asarray(angles, dtype=float)
        n = len(chain.links)
        if arr.size == n:
            return arr
        # try chain.full_from_active (many ikpy versions provide it)
        try:
            full = chain.full_from_active(arr)
            full = np.asarray(full, dtype=float)
            if full.size == n:
                return full
        except Exception:
            pass
        # try interpreting as active-from-full fallback: pad active into default full
        try:
            active_mask = getattr(chain, "active_links_mask", None)
            if active_mask is not None and len(active_mask) == n:
                full = np.zeros(n, dtype=float)
                ai = 0
                for i, m in enumerate(active_mask):
                    if bool(m):
                        if ai < arr.size:
                            full[i] = arr[ai]
                        ai += 1
                return full
        except Exception:
            pass
        # final fallback: pad/truncate
        if arr.size < n:
            return np.concatenate([arr, np.zeros(n - arr.size)])
        return arr[:n]

    # format angles for display (degrees, short)
    def format_angles(angles):
        try:
            a = np.asarray(angles, dtype=float)
            degs = np.rad2deg(a)
            return ", ".join(f"{v:.1f}°" for v in degs)
        except Exception:
            return "n/a"

    # cleaned draw_chains uses chain.plot and ensures full-length angles
    def draw_chains(angles_left_arm, angles_right_arm, target_left_pt, target_right_pt):
        ax.cla()
        ax.set_xlim(-0.5,0.5); ax.set_ylim(-0.5,0.5); ax.set_zlim(0,0.8)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        # normalize angle vectors
        ala = _ensure_full_angles(robotis_op3_left_arm_chain, angles_left_arm)
        ara = _ensure_full_angles(robotis_op3_right_arm_chain, angles_right_arm)
        # left arm
        try:
            robotis_op3_left_arm_chain.plot(ala, ax, target=target_left_pt)
        except Exception as e:
            print("Draw left arm error:", e)
        # right arm
        try:
            robotis_op3_right_arm_chain.plot(ara, ax, target=target_right_pt)
        except Exception as e:
            print("Draw right arm error:", e)
        # legs static (plot full zeros sized to each chain)
        try:
            robotis_op3_left_leg_chain.plot(np.zeros(len(robotis_op3_left_leg_chain.links)), ax)
        except Exception:
            pass
        try:
            robotis_op3_right_leg_chain.plot(np.zeros(len(robotis_op3_right_leg_chain.links)), ax)
        except Exception:
            pass
        # target markers
        if target_left_pt is not None:
            try:
                ax.scatter(*target_left_pt, color="red", s=50, label="Target left")
            except Exception:
                pass
        if target_right_pt is not None:
            try:
                ax.scatter(*target_right_pt, color="blue", s=50, label="Target right")
            except Exception:
                pass
        ax.legend()
        try:
            canvas.draw_idle()
        except Exception:
            pass

    # initial draw using zeros sized to each chain
    draw_chains(
        np.zeros(len(robotis_op3_left_arm_chain.links)),
        np.zeros(len(robotis_op3_right_arm_chain.links)),
        [0,0,0],
        [0,0,0]
    )

    # Define a non-zero initial pose to help the IK solver
    # This can be tuned for a better "default" stance.
    # The values are in radians. This pose has the elbows slightly bent.
    initial_pose_left = np.zeros(len(robotis_op3_left_arm_chain.links))
    if len(initial_pose_left) > 4:
        initial_pose_left[4] = -0.5  # left_elbow_joint

    initial_pose_right = np.zeros(len(robotis_op3_right_arm_chain.links))
    if len(initial_pose_right) > 4:
        initial_pose_right[4] = -0.5 # right_elbow_joint

    def update_loop():
        if running["val"]:
            mode = control_mode.get()
            
            if mode == "IK":
                ltgt = [float(lx.get()), float(ly.get()), float(lz.get())]
                rtgt = [float(rx.get()), float(ry.get()), float(rz.get())]

                # calculate IK for left and right arms using robust helper
                ik_left = safe_inverse_kinematics(robotis_op3_left_arm_chain, ltgt, initial_position=initial_pose_left)
                ik_right = safe_inverse_kinematics(robotis_op3_right_arm_chain, rtgt, initial_position=initial_pose_right)

                # ensure full-length before plotting and display
                ik_left_full = _ensure_full_angles(robotis_op3_left_arm_chain, ik_left)
                ik_right_full = _ensure_full_angles(robotis_op3_right_arm_chain, ik_right)
                
                # update angle display (show active joints only)
                try:
                    left_active = robotis_op3_left_arm_chain.active_from_full(ik_left_full)
                except Exception:
                    left_active = ik_left_full
                try:
                    right_active = robotis_op3_right_arm_chain.active_from_full(ik_right_full)
                except Exception:
                    right_active = ik_right_full

                angles_text.delete("1.0", tk.END)
                angles_text.insert(tk.END, "Left arm angles: \n" + format_angles(left_active) + "\n\n")
                angles_text.insert(tk.END, "Right arm angles:\n" + format_angles(right_active) + "\n")

                # draw
                draw_chains(ik_left_full, ik_right_full, ltgt, rtgt)

            elif mode == "FK":
                # Get angles from sliders (in degrees) and convert to radians
                left_active_angles_rad = [np.deg2rad(v.get()) for v in left_joint_vars]
                right_active_angles_rad = [np.deg2rad(v.get()) for v in right_joint_vars]

                # Convert active angles to full angle vectors
                ik_left_full = robotis_op3_left_arm_chain.inverse_kinematics([0,0,0], initial_position=np.zeros(len(robotis_op3_left_arm_chain.links))) # get a zero vector of correct size
                ik_right_full = robotis_op3_right_arm_chain.inverse_kinematics([0,0,0], initial_position=np.zeros(len(robotis_op3_right_arm_chain.links)))

                # Fill in the active joint values
                left_active_indices = [i for i, active in enumerate(robotis_op3_left_arm_chain.active_links_mask) if active]
                for i, angle in zip(left_active_indices, left_active_angles_rad):
                    ik_left_full[i] = angle
                
                right_active_indices = [i for i, active in enumerate(robotis_op3_right_arm_chain.active_links_mask) if active]
                for i, angle in zip(right_active_indices, right_active_angles_rad):
                    ik_right_full[i] = angle

                # Calculate FK to find end-effector positions
                fk_left_matrix = robotis_op3_left_arm_chain.forward_kinematics(ik_left_full)
                fk_right_matrix = robotis_op3_right_arm_chain.forward_kinematics(ik_right_full)
                ltgt = fk_left_matrix[:3, 3]
                rtgt = fk_right_matrix[:3, 3]

                # Sync IK sliders with FK-derived positions
                lx.set(ltgt[0])
                ly.set(ltgt[1])
                lz.set(ltgt[2])
                rx.set(rtgt[0])
                ry.set(rtgt[1])
                rz.set(rtgt[2])

                angles_text.delete("1.0", tk.END)
                angles_text.insert(tk.END, "Left arm angles: \n" + format_angles(left_active_angles_rad) + "\n\n")
                angles_text.insert(tk.END, "Right arm angles:\n" + format_angles(right_active_angles_rad) + "\n")

                # draw
                draw_chains(ik_left_full, ik_right_full, ltgt, rtgt)

        root.after(update_interval_ms, update_loop)

    # start loop
    root.after(100, update_loop)
    root.mainloop()
        # update plot and angles text

