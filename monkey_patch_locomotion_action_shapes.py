# monkey_patch_locomotion_action_shapes.py
# แพตช์ LocomotionEnv._apply_action ให้จัด shape ของ action_scale / joint_gears ถูกต้อง
# ใช้แบบ: หลัง AppLauncher สตาร์ทแล้ว ให้เรียก apply_patch()

from typing import Any
import types
import torch
import importlib

def _as_1x_actdim(x: Any, act_dim: int, device: torch.device, name: str) -> torch.Tensor:
    """
    แปลงค่าคงที่/ลิสต์/เทนเซอร์ให้เป็นทรง (1, act_dim)
    - สเกลาร์ -> expand เป็น (act_dim,)
    - ลิสต์/เทนเซอร์ความยาว act_dim -> reshape เป็น (1, act_dim)
    """
    t = torch.as_tensor(x, device=device, dtype=torch.float32)
    if t.ndim == 0:
        t = t.expand(act_dim)  # scalar -> (act_dim,)
    elif t.numel() != act_dim:
        raise RuntimeError(f"{name} length must be {act_dim}, got {t.numel()}")
    return t.view(1, act_dim)  # (1, act_dim)

def _ensure_abcast_params(self) -> None:
    """
    เตรียม self.action_scale, self.joint_gears ให้อยู่ทรง (1, act_dim) เพื่อ broadcast กับ actions:(num_envs, act_dim)
    รันครั้งเดียวและ cache ไว้
    """
    if getattr(self, "_ab_params_ready", False):
        return

    device = self.device if hasattr(self, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- หา act_dim ให้ถูกต้อง: ใช้จาก self.actions เป็นอันดับแรก ---
    act_dim = None
    if hasattr(self, "actions") and isinstance(self.actions, torch.Tensor) and self.actions.ndim >= 2:
        act_dim = int(self.actions.shape[-1])
    if act_dim is None and hasattr(self, "num_actions"):
        act_dim = int(self.num_actions)
    if act_dim is None and hasattr(self, "action_space") and hasattr(self.action_space, "shape"):
        shp = tuple(self.action_space.shape)
        act_dim = int(shp[-1])
    if act_dim is None:
        raise RuntimeError("Cannot infer action dimension (act_dim).")

    # ดึงค่าจาก cfg ถ้ามี ไม่มีก็ fallback เป็น 1.0
    cfg = getattr(self, "cfg", None)
    cfg_action_scale = getattr(cfg, "action_scale", 1.0) if cfg is not None else 1.0
    cfg_joint_gears = getattr(cfg, "joint_gears", 1.0) if cfg is not None else 1.0

    # แปลงเป็น (1, act_dim)
    self.action_scale = _as_1x_actdim(cfg_action_scale, act_dim, device, "action_scale")
    self.joint_gears = _as_1x_actdim(cfg_joint_gears, act_dim, device, "joint_gears")

    self._ab_params_ready = True

    # ดีบั๊กครั้งแรก
    if not getattr(self, "_ab_params_logged", False):
        print(f"[ABCAST] act_dim={act_dim} "
              f"actions.shape={(tuple(self.actions.shape) if hasattr(self,'actions') and isinstance(self.actions, torch.Tensor) else None)} "
              f"action_scale.shape={tuple(self.action_scale.shape)} "
              f"joint_gears.shape={tuple(self.joint_gears.shape)}")
        self._ab_params_logged = True

def _patched_apply_action(self) -> None:
    """
    เวอร์ชันแพตช์ของ _apply_action:
    - ensure shape: actions:(N, A), action_scale:(1, A), joint_gears:(1, A)
    - คูณแบบ elementwise ปลอดภัย
    """
    if not hasattr(self, "actions") or not isinstance(self.actions, torch.Tensor) or self.actions.ndim < 2:
        raise RuntimeError("self.actions must be a 2D tensor (num_envs, act_dim) before applying action.")

    _ensure_abcast_params(self)

    actions = self.actions  # (N, A)
    forces = actions * self.action_scale * self.joint_gears  # (N,A) * (1,A) * (1,A) -> (N,A)

    # ส่งไปยัง impl เดิมถ้ามี ไม่งั้นเก็บไว้ให้ step() ใช้ต่อ
    if hasattr(self, "_apply_action_impl") and isinstance(self._apply_action_impl, types.MethodType):
        self._apply_action_impl(forces)
    else:
        self._last_forces = forces

def apply_patch():
    """
    เรียกหลัง AppLauncher app สตาร์ทแล้วเท่านั้น!
    """
    locomotion_mod = importlib.import_module("isaaclab_tasks.direct.locomotion.locomotion_env")
    LocomotionEnv = getattr(locomotion_mod, "LocomotionEnv")

    # เก็บของเดิมไว้เผื่อ revert
    if not hasattr(LocomotionEnv, "_apply_action_original"):
        setattr(LocomotionEnv, "_apply_action_original", LocomotionEnv._apply_action)

    # ผูกเมธอดแพตช์
    LocomotionEnv._apply_action = _patched_apply_action

    # ใส่ util สำหรับ debug (optional)
    LocomotionEnv._ensure_abcast_params = _ensure_abcast_params
