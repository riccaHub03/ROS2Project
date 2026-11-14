"""IK differenziale (damped least squares) con vincoli di velocita', accelerazione e jerk dei giunti.

Sostituisce il controllo "a posteriori" con un controllore che *rispetta* i limiti a ogni passo:

  * Jacobiano numerico del TCP + pseudoinversa smorzata (smorzamento adattivo vicino alle singolarita').
  * Il moto nello spazio nullo tiene la postura vicina a quella di partenza (evita i cambi di ramo).
  * I limiti diventano **bound sul passo** ``dq``: velocita', accelerazione, jerk e posizione (con frenata
    anticipata) si intersecano in un intervallo ``[lo, hi]`` per giunto, ricalcolato a ogni tick.
  * Il tracking e' un *path following*: il riferimento avanza lungo il segmento cartesiano con un
    parametro ``s``. Se i bound non consentono di tenere la velocita' nominale, ``s`` rallenta (override
    in forma chiusa, la direzione del moto non cambia); se serve altro, un solver a insieme attivo risolve
    il problema ai minimi quadrati con bound.

Il nucleo e' numpy puro: la cinematica e' un oggetto con ``fk(q) -> (posizione, matrice_rotazione)``.
``PyBulletKinematics`` ne e' l'adattatore per la scena (client DIRECT separato, la fisica non viene toccata).

Tutto e' attivabile da ``motion_control.differential_ik.enabled`` nel YAML; spento, non cambia nulla.
La traiettoria e' calcolata in anticipo (offline) e riprodotta da ``MotionPreview.tick``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from pipeline.robot_model import RobotModel

CONTROL_DT = 1 / 240
DEFAULT_PLOT = "sim_robot_joint_velocities.png"
DEFAULT_PLOT_DYNAMICS = "sim_robot_joint_acc_jerk.png"
# Joint velocities come from the shared robot URDF. Acceleration and jerk below
# are configured simulation bounds, not manufacturer-certified hardware values.
DEFAULT_MAX_ACCELERATION_RAD_S2 = 4.0
DEFAULT_MAX_JERK_RAD_S3 = 40.0


# ---------------------------------------------------------------------------------------------------
# Rotazioni
# ---------------------------------------------------------------------------------------------------
def rot_exp(vector) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = np.asarray(vector, float) / angle
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def rot_log(R) -> np.ndarray:
    angle = math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:                       # vicino a pi: l'asse viene dalla parte simmetrica
        B = (R + np.eye(3)) / 2
        axis = np.sqrt(np.maximum(np.diag(B), 0))
        k = int(np.argmax(axis))
        axis = B[:, k] / max(axis[k], 1e-12)
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return axis * angle


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------------------------------------------
# Algebra: pseudoinversa smorzata e minimi quadrati con bound
# ---------------------------------------------------------------------------------------------------
def dls_pinv(J, damping_max: float, singularity_threshold: float):
    """Pseudoinversa smorzata. Lo smorzamento cresce solo quando il valore singolare minimo e' piccolo."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    smallest = float(S.min())
    lam2 = 0.0 if smallest >= singularity_threshold else (1 - (smallest / singularity_threshold) ** 2) * damping_max ** 2
    return (Vt.T * (S / (S ** 2 + lam2))) @ U.T, smallest


def null_projector(J, tolerance: float = 1e-4) -> np.ndarray:
    """Proiettore esatto sullo spazio nullo (non smorzato): il moto secondario non disturba il compito."""
    _, S, Vt = np.linalg.svd(J)
    V = Vt[:int((S > tolerance).sum())].T
    return np.eye(J.shape[1]) - V @ V.T


def bounded_dls(J, task, nullspace_dq, lo, hi, damping_max, singularity_threshold):
    """min ||J dq - task||^2 (smorzato) con lo <= dq <= hi, a insieme attivo.

    I giunti che violano un bound vengono fissati al bound e il compito residuo e' ridistribuito sugli altri.
    """
    n = J.shape[1]
    fixed = np.zeros(n, bool)
    dq = np.zeros(n)
    for _ in range(n):
        free = ~fixed
        if not free.any():
            break
        Jf = J[:, free]
        Jp, _ = dls_pinv(Jf, damping_max, singularity_threshold)
        residual = task - J[:, fixed] @ dq[fixed]
        dq[free] = Jp @ residual + (np.eye(int(free.sum())) - Jp @ Jf) @ nullspace_dq[free]
        violation = np.maximum(lo - dq, dq - hi)
        violation[fixed] = 0
        worst = int(violation.argmax())
        if violation[worst] <= 1e-12:
            break
        dq[worst] = np.clip(dq[worst], lo[worst], hi[worst])
        fixed[worst] = True
    return np.clip(dq, lo, hi)


# ---------------------------------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------------------------------
def _per_joint(value, n, name):
    values = [float(value)] * n if np.isscalar(value) else [float(v) for v in value]
    if len(values) != n or min(values) <= 0:
        raise ValueError(f"differential_ik.{name}: servono {n} valori positivi")
    return np.asarray(values)


@dataclass
class DifferentialIKConfig:
    enabled: bool = False
    max_velocity: np.ndarray = field(default_factory=lambda: np.asarray(RobotModel.from_config().velocity))
    max_acceleration: np.ndarray = field(default_factory=lambda: np.full(7, DEFAULT_MAX_ACCELERATION_RAD_S2))
    max_jerk: np.ndarray = field(default_factory=lambda: np.full(7, DEFAULT_MAX_JERK_RAD_S3))
    velocity_safety_factor: float = 0.6
    position_margin_rad: float = 0.05
    damping_max: float = 0.05
    singularity_threshold: float = 0.05
    nullspace_gain: float = 1.0
    feedback_gain: float = 5.0
    cartesian_linear_speed_m_s: float = 0.12
    cartesian_angular_speed_rad_s: float = 0.8
    min_phase_duration_s: float = 0.6
    position_tolerance_m: float = 0.003      # come solve_ik
    orientation_tolerance_rad: float = 0.035
    plot: bool = True
    plot_filename: str = DEFAULT_PLOT
    plot_dynamics_filename: str = DEFAULT_PLOT_DYNAMICS

    @classmethod
    def from_config(cls, config: dict, joints: int = 7) -> "DifferentialIKConfig":
        motion = (config or {}).get("motion_control", {}) or {}
        raw = motion.get("differential_ik", {}) or {}
        if not raw.get("enabled", False):
            return cls(enabled=False)
        hardware_velocity = np.asarray(RobotModel.from_config(config).velocity)
        requested_velocity = raw.get("joint_max_velocity_rad_s")
        if requested_velocity is None:
            requested_velocity = hardware_velocity
        defaults = cls()
        simple = {k: raw[k] for k in (
            "velocity_safety_factor", "position_margin_rad", "nullspace_gain", "feedback_gain",
            "min_phase_duration_s",
            "position_tolerance_m", "orientation_tolerance_rad", "plot", "plot_filename",
            "plot_dynamics_filename") if k in raw}
        damping = raw.get("damping", {}) or {}
        velocities = motion.get("velocities", {}) or {}
        config_obj = cls(
            enabled=True,
            max_velocity=np.minimum(_per_joint(requested_velocity, joints,
                                               "joint_max_velocity_rad_s"), hardware_velocity),
            max_acceleration=_per_joint(raw.get("joint_max_acceleration_rad_s2", DEFAULT_MAX_ACCELERATION_RAD_S2),
                                        joints, "joint_max_acceleration_rad_s2"),
            max_jerk=_per_joint(raw.get("joint_max_jerk_rad_s3", DEFAULT_MAX_JERK_RAD_S3), joints,
                                "joint_max_jerk_rad_s3"),
            cartesian_linear_speed_m_s=float(velocities.get("cartesian_linear_speed_m_s",
                                                            defaults.cartesian_linear_speed_m_s)),
            cartesian_angular_speed_rad_s=float(velocities.get("cartesian_angular_speed_rad_s",
                                                               defaults.cartesian_angular_speed_rad_s)),
            damping_max=float(damping.get("max", defaults.damping_max)),
            singularity_threshold=float(damping.get("singularity_threshold", defaults.singularity_threshold)),
            **simple)
        if not 0 < config_obj.velocity_safety_factor <= 1:
            raise ValueError("differential_ik.velocity_safety_factor deve stare in (0, 1]")
        return config_obj


# ---------------------------------------------------------------------------------------------------
# Controllore
# ---------------------------------------------------------------------------------------------------
class ConstrainedDifferentialIK:
    def __init__(self, fk: Callable, q_min, q_max, config: DifferentialIKConfig, q0, dt: float = CONTROL_DT):
        self.fk, self.cfg, self.dt = fk, config, dt
        self.q_min, self.q_max = np.asarray(q_min, float), np.asarray(q_max, float)
        self.n = len(self.q_min)
        self.q = np.asarray(q0, float).copy()
        self.q_rest = self.q.copy()                   # il moto nello spazio nullo richiama questa postura
        self.dq = np.zeros(self.n)
        self.ddq = np.zeros(self.n)
        self.phase_stats: list[dict] = []
        self.log: dict[str, list] = {k: [] for k in ("t", "phase", "requested", "applied", "acc", "jerk", "override")}
        self._clock = 0.0
        self._violations = 0

    # -- cinematica -------------------------------------------------------------------------------------
    def pose_and_jacobian(self, q):
        position, R = self.fk(q)
        position = np.asarray(position, float)
        J = np.zeros((6, self.n))
        h = 1e-5
        for j in range(self.n):
            shifted = q.copy()
            shifted[j] += h
            p1, R1 = self.fk(shifted)
            J[:3, j] = (np.asarray(p1) - position) / h
            J[3:, j] = rot_log(R1 @ R.T) / h
        return position, R, J

    # -- bound sul passo ------------------------------------------------------------------------------------
    def hard_bounds(self):
        """Velocita' e posizione (con frenata anticipata): sicurezza, hanno sempre la precedenza."""
        c, dt = self.cfg, self.dt
        vmax = c.max_velocity * c.velocity_safety_factor
        room_hi = np.maximum(self.q_max - c.position_margin_rad - self.q, 0.0)
        room_lo = np.maximum(self.q - (self.q_min + c.position_margin_rad), 0.0)
        brake = np.sqrt(2 * 0.5 * c.max_acceleration)    # ci si deve poter fermare entro il margine con A/2
        hi = np.minimum(vmax, np.minimum(room_hi / dt, brake * np.sqrt(room_hi)))
        lo = -np.minimum(vmax, np.minimum(room_lo / dt, brake * np.sqrt(room_lo)))
        return lo, hi

    def step_bounds(self, hard_lo, hard_hi):
        """Intervallo ammesso per il prossimo dq: hard ∩ (accelerazione ∩ jerk)."""
        c, dt = self.cfg, self.dt
        acc = c.max_acceleration * dt
        center = self.dq + self.ddq * dt
        jerk = c.max_jerk * dt * dt
        soft_lo = np.maximum(self.dq - acc, center - jerk)
        soft_hi = np.minimum(self.dq + acc, center + jerk)
        lo, hi = np.maximum(hard_lo, soft_lo), np.minimum(hard_hi, soft_hi)
        clash = lo > hi
        if clash.any():                                  # sicurezza prima di acc/jerk
            value = np.clip((soft_lo + soft_hi) / 2, hard_lo, hard_hi)
            lo, hi = np.where(clash, value, lo), np.where(clash, value, hi)
        return lo, hi

    def shape_velocity(self, dq_target):
        """Inseguitore di velocita' con jerk e accelerazione limitati, senza overshoot.

        Accelerazione obiettivo = sign(e) * min(A, sqrt(2 J |e|)), dove e e' l'errore di velocita' *previsto*
        quando l'accelerazione corrente sara' riportata a zero; il jerk la raggiunge in rampa.
        """
        c, dt = self.cfg, self.dt
        predicted = self.dq + np.sign(self.ddq) * self.ddq ** 2 / (2 * c.max_jerk)
        error = dq_target - predicted
        target_acc = np.sign(error) * np.minimum(c.max_acceleration, np.sqrt(2 * c.max_jerk * np.abs(error)))
        ddq = self.ddq + np.clip(target_acc - self.ddq, -c.max_jerk * dt, c.max_jerk * dt)
        return self.dq + ddq * dt

    # -- una fase di moto --------------------------------------------------------------------------------
    def plan_phase(self, label: str, target_position, target_R, nominal_duration: Optional[float] = None,
                   deadline: Optional[float] = None) -> np.ndarray:
        """Genera i campioni q a passo ``dt`` dalla posa corrente alla posa target, ferma a inizio e fine."""
        c, dt = self.cfg, self.dt
        target_position = np.asarray(target_position, float)
        p0, R0 = self.fk(self.q)
        p0 = np.asarray(p0, float)
        delta_p = target_position - p0
        w_rel = rot_log(R0.T @ target_R)
        if nominal_duration is None:
            nominal_duration = max(c.min_phase_duration_s, np.linalg.norm(delta_p) / c.cartesian_linear_speed_m_s,
                                   np.linalg.norm(w_rel) / c.cartesian_angular_speed_rad_s)
        s_speed_max = 1.0 / nominal_duration
        s_accel = s_speed_max / 0.8                      # rampa cartesiana di circa 0.8 s
        s, s_rate = 0.0, 0.0
        samples = [self.q.copy()]
        override_min, steps, limit_hits = 1.0, 0, 0
        position_error = orientation_error = float('inf')
        settling = False
        max_steps = int((nominal_duration * 8 + 5) / dt)
        for steps in range(1, max_steps + 1):
            if deadline is not None and steps % 16 == 0 and time.monotonic() > deadline:
                raise ValueError(f"{label}: candidate planning time budget exceeded")
            position, R, J = self.pose_and_jacobian(self.q)
            pinv, _ = dls_pinv(J, c.damping_max, c.singularity_threshold)
            null = null_projector(J)
            ref_position = p0 + s * delta_p
            ref_R = R0 @ rot_exp(s * w_rel)
            error = np.concatenate([ref_position - position, rot_log(ref_R @ R.T)])
            s_rate_nominal = min(s_speed_max, s_rate + s_accel * dt, math.sqrt(2 * s_accel * max(1 - s, 0.0)))
            if s < 1:
                s_rate_nominal = max(s_rate_nominal, 0.03 * s_speed_max)
            feedforward = np.concatenate([delta_p, R0 @ w_rel]) * s_rate_nominal
            nullspace_dq = c.nullspace_gain * (1 - s) * (self.q_rest - self.q)      # si spegne a fine fase: si arriva fermi
            feedback = c.feedback_gain * error
            a_free = pinv @ feedback + null @ nullspace_dq
            b = pinv @ feedforward
            hard_lo, hard_hi = self.hard_bounds()
            # 1) correzione di tracking: minimi quadrati con bound (== a_free se nessun limite e' attivo)
            if np.all((a_free >= hard_lo - 1e-12) & (a_free <= hard_hi + 1e-12)):
                a = a_free
            else:
                a = bounded_dls(J, feedback, nullspace_dq, hard_lo, hard_hi, c.damping_max, c.singularity_threshold)
                limit_hits += 1
            # 2) avanzamento: il feedforward cresce fino al primo bound di velocita'/posizione (forma chiusa)
            with np.errstate(divide="ignore", invalid="ignore"):
                room = np.where(b > 1e-12, (hard_hi - a) / b, np.where(b < -1e-12, (hard_lo - a) / b, np.inf))
            override = float(np.clip(room.min(), 0.0, 1.0))
            # 3) accelerazione e jerk: inseguitore di velocita' liscio, poi intervallo ammesso come rete finale
            lo, hi = self.step_bounds(hard_lo, hard_hi)
            desired = np.zeros(self.n) if settling else a + override * b      # a posa raggiunta: frenata fino a fermo
            applied = np.clip(self.shape_velocity(desired), lo, hi)
            requested = a_free + b

            new_ddq = (applied - self.dq) / dt
            self.log["t"].append(self._clock)
            self.log["phase"].append(label)
            self.log["requested"].append(requested)
            self.log["applied"].append(applied.copy())
            self.log["acc"].append(new_ddq)
            self.log["jerk"].append((new_ddq - self.ddq) / dt)
            self.log["override"].append(override)
            self._clock += dt
            override_min = min(override_min, override)
            self.q = self.q + applied * dt
            self.dq, self.ddq = applied, new_ddq
            s_rate = s_rate_nominal * override
            s = min(1.0, s + s_rate * dt)
            samples.append(self.q.copy())
            if s >= 1.0:
                p_now, R_now = self.fk(self.q)
                position_error = float(np.linalg.norm(target_position - p_now))
                orientation_error = float(np.linalg.norm(rot_log(target_R @ R_now.T)))
                settling = settling or (position_error < c.position_tolerance_m
                                        and orientation_error < c.orientation_tolerance_rad)
                if settling and float(np.abs(self.dq).max()) < 2e-3:
                    break
        else:
            if s >= 1.0:
                raise ValueError(f"{label}: posa non raggiunta entro i limiti di velocita'/accelerazione/jerk "
                                 f"(errore {position_error * 1000:.1f} mm, {math.degrees(orientation_error):.1f} gradi)")
            raise ValueError(f"{label}: avanzamento bloccato ai limiti dei giunti (probabile singolarita')")
        self.dq, self.ddq = np.zeros(self.n), np.zeros(self.n)
        self.phase_stats.append({
            "phase": label, "duration_s": steps * dt, "nominal_duration_s": nominal_duration,
            "time_scale": steps * dt / nominal_duration, "min_override": override_min,
            "bounded_solver_ticks": limit_hits,
        })
        return np.asarray(samples)

    # -- report --------------------------------------------------------------------------------------------
    def summary(self) -> dict:
        c = self.cfg
        if not self.log["t"]:
            return {"enabled": True, "phases": self.phase_stats}
        peak = {k: np.abs(np.asarray(self.log[k])).max(axis=0) for k in ("requested", "applied", "acc", "jerk")}
        eps = 1e-6
        return {
            "enabled": True, "method": "dls_velocity_position_bounds_jerk_limited_tracker",
            "joint_max_velocity_rad_s": c.max_velocity.tolist(),
            "joint_max_acceleration_rad_s2": c.max_acceleration.tolist(),
            "joint_max_jerk_rad_s3": c.max_jerk.tolist(),
            "velocity_safety_factor": c.velocity_safety_factor,
            "peak_requested_velocity_rad_s": peak["requested"].tolist(),
            "peak_applied_velocity_rad_s": peak["applied"].tolist(),
            "peak_applied_acceleration_rad_s2": peak["acc"].tolist(),
            "peak_applied_jerk_rad_s3": peak["jerk"].tolist(),
            "applied_within_limits": bool(
                np.all(peak["applied"] <= c.max_velocity * c.velocity_safety_factor + eps)
                and np.all(peak["acc"] <= c.max_acceleration + eps)
                and np.all(peak["jerk"] <= c.max_jerk + eps)),
            "phases": self.phase_stats,
        }

    def save_plot(self, directory) -> dict:
        """Scrive i PNG: velocita' richiesta vs applicata (7 giunti) e accelerazione/jerk."""
        if not (self.cfg.plot and self.log["t"]):
            return {}
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {}
        c, log = self.cfg, self.log
        t = np.asarray(log["t"])
        requested, applied = np.asarray(log["requested"]), np.asarray(log["applied"])
        starts = [t[i] for i in range(len(t)) if i == 0 or log["phase"][i] != log["phase"][i - 1]]
        vlimit = c.max_velocity
        figure, axes = plt.subplots(4, 2, figsize=(13, 10))
        flat = axes.ravel()
        for j in range(requested.shape[1]):
            ax = flat[j]
            ax.plot(t, requested[:, j], color="#9aa5b1", lw=0.8, label="richiesta (IK senza vincoli)")
            ax.plot(t, applied[:, j], color="#1f6feb", lw=1.4, label="applicata (con vincoli)")
            for sign in (1, -1):
                ax.axhline(sign * vlimit[j], color="#d1242f", ls="--", lw=0.9,
                           label="limite hardware" if sign == 1 else None)
                ax.axhline(sign * vlimit[j] * c.velocity_safety_factor, color="#d29922", ls=":", lw=0.9,
                           label="limite di progetto" if sign == 1 else None)
            for start in starts:
                ax.axvline(start, color="#d0d7de", lw=0.6)
            ax.set_title(f"Giunto {j + 1}", fontsize=10)
            ax.set_ylabel("rad/s")
            ax.set_xlabel("tempo di moto [s]")
            ax.set_ylim(-1.15 * vlimit[j], 1.15 * vlimit[j])
        flat[0].legend(fontsize=7, loc="upper right")
        last = flat[7]
        last.plot(t, log["override"], color="#1f6feb", lw=1.2)
        last.set_ylim(0, 1.05)
        last.set_title("Override di avanzamento (1 = velocita' nominale)", fontsize=10)
        last.set_xlabel("tempo di moto [s]")
        figure.suptitle("Velocita' dei giunti: richiesta vs applicata", fontsize=12)
        figure.tight_layout()
        out = {"velocity": str(Path(directory) / c.plot_filename)}
        figure.savefig(out["velocity"], dpi=110)
        plt.close(figure)

        figure, (ax_a, ax_j) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        acc, jerk = np.asarray(log["acc"]), np.asarray(log["jerk"])
        for j in range(acc.shape[1]):
            ax_a.plot(t, acc[:, j], lw=0.9, label=f"giunto {j + 1}")
            ax_j.plot(t, jerk[:, j], lw=0.9)
        amax, jmax = float(c.max_acceleration.max()), float(c.max_jerk.max())
        for ax, limit, name in ((ax_a, amax, "rad/s²"), (ax_j, jmax, "rad/s³")):
            ax.axhline(limit, color="#d1242f", ls="--", lw=0.9)
            ax.axhline(-limit, color="#d1242f", ls="--", lw=0.9)
            ax.set_ylabel(name)
            for start in starts:
                ax.axvline(start, color="#d0d7de", lw=0.6)
        ax_a.set_title("Accelerazione applicata (tratteggio = limite)")
        ax_j.set_title("Jerk applicato (tratteggio = limite)")
        ax_j.set_xlabel("tempo di moto [s]")
        ax_a.legend(fontsize=7, ncol=7, loc="upper right")
        figure.tight_layout()
        out["dynamics"] = str(Path(directory) / c.plot_dynamics_filename)
        figure.savefig(out["dynamics"], dpi=110)
        plt.close(figure)
        return out


# ---------------------------------------------------------------------------------------------------
# Adattatore PyBullet
# ---------------------------------------------------------------------------------------------------
class PyBulletKinematics:
    """FK del TCP su una copia del robot in un client DIRECT separato: la fisica della scena non cambia."""

    def __init__(self, urdf, base_position, base_quaternion, joints, ee, tcp_offset):
        import pybullet as p
        import pybullet_data
        self._p = p
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.robot = p.loadURDF(urdf, basePosition=list(base_position), baseOrientation=list(base_quaternion),
                                useFixedBase=True, physicsClientId=self.cid)
        self.joints, self.ee, self.tcp_offset = list(joints), ee, list(tcp_offset)

    def fk(self, q):
        p = self._p
        for joint, value in zip(self.joints, q):
            p.resetJointState(self.robot, joint, float(value), physicsClientId=self.cid)
        state = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True, physicsClientId=self.cid)
        position, orientation = p.multiplyTransforms(state[4], state[5], self.tcp_offset, [0, 0, 0, 1])
        return np.asarray(position), np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

    def close(self):
        cid, self.cid = self.cid, None
        if cid is not None and self._p.isConnected(cid):
            self._p.disconnect(cid)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_for_scene(scene) -> Optional[tuple[ConstrainedDifferentialIK, PyBulletKinematics]]:
    """Costruisce controllore e cinematica per la scena, o ``None`` se disattivato nel YAML."""
    import pybullet as p
    config = DifferentialIKConfig.from_config(scene.config, len(scene.joints))
    if not config.enabled:
        return None
    kinematics = PyBulletKinematics(str(scene.robot_model.urdf), scene.base_frame.position, scene.base_frame.quaternion,
                                    scene.joints, scene.ee, scene.gripper_geometry.tcp_offset)
    q0 = [p.getJointState(scene.robot, j, physicsClientId=scene.cid)[0] for j in scene.joints]
    limits = np.asarray(scene.limits, float)
    return ConstrainedDifferentialIK(kinematics.fk, limits[:, 0], limits[:, 1], config, q0), kinematics
