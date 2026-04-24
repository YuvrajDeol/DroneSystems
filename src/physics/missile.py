from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MissileState:
    """2D point-mass state in (range, altitude) plane.

    Units:
    - position: meters
    - velocity: meters/second
    """

    position_m: np.ndarray  # shape (2,)
    velocity_mps: np.ndarray  # shape (2,)


@dataclass(slots=True)
class Missile:
    mass_kg: float = 100.0
    max_accel_mps2: float = 30.0
    air_density_kgpm3: float = 1.225  # sea-level nominal density
    drag_area_m2: float = 0.05  # effective Cd * A
    cruise_speed_mps: float = 850.0
    speed_hold_kp: float = 0.03
    max_thrust_n: float = 12000.0
    gravity_mps2: float = 9.81

    def _clamp_accel_cmd(self, accel_cmd_mps2: np.ndarray) -> np.ndarray:
        accel_cmd = np.asarray(accel_cmd_mps2, dtype=float).reshape(2).copy()
        cmd_norm = float(np.linalg.norm(accel_cmd))
        if cmd_norm > self.max_accel_mps2 and cmd_norm > 0.0:
            accel_cmd *= self.max_accel_mps2 / cmd_norm
        return accel_cmd

    def _get_air_density(self, altitude_m: float) -> float:
        alt_non_negative = max(0.0, altitude_m)
        return float(self.air_density_kgpm3 * np.exp(-alt_non_negative / 8500.0))

    def _get_speed_of_sound(self, altitude_m: float) -> float:
        """ISA-like model with stratosphere floor."""
        alt_non_negative = max(0.0, altitude_m)
        return float(max(295.0, 340.29 - 0.00648 * alt_non_negative))

    def _get_drag_area(self, speed_mps: float, altitude_m: float) -> float:
        # NOTE: Simplified transonic drag spike for initial validation.
        a_local = self._get_speed_of_sound(altitude_m)
        mach = speed_mps / max(a_local, 1e-6)

        if mach < 0.8:
            mult = 1.0
        elif mach < 1.1:
            # Rise from 1.0 at M0.8 to 2.5 at M1.1.
            mult = 1.0 + (mach - 0.8) * (1.5 / 0.3)
        elif mach < 2.5:
            # Decay from 2.5 at M1.1 to 1.5 at M2.5.
            mult = 2.5 - (mach - 1.1) * (1.0 / 1.4)
        else:
            mult = 1.5

        return float(self.drag_area_m2 * mult)

    def _compute_derivatives(
        self, state: MissileState, accel_cmd_mps2: np.ndarray, hold_cruise_speed: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        vel = state.velocity_mps
        altitude_m = float(state.position_m[1])
        speed = float(np.linalg.norm(vel))
        if speed > 1e-6:
            vel_hat = vel / speed
        else:
            vel_hat = np.array([1.0, 0.0], dtype=float)

        rho = self._get_air_density(altitude_m)
        drag_area = self._get_drag_area(speed_mps=speed, altitude_m=altitude_m)
        drag_n = 0.5 * rho * speed * speed * drag_area
        drag_accel_vec = -(drag_n / self.mass_kg) * vel_hat

        thrust_n = 0.0
        if hold_cruise_speed:
            speed_error = self.cruise_speed_mps - speed
            thrust_cmd = drag_n + self.speed_hold_kp * self.mass_kg * speed_error
            thrust_n = float(np.clip(thrust_cmd, 0.0, self.max_thrust_n))
        thrust_accel_vec = (thrust_n / self.mass_kg) * vel_hat

        gravity_vec = np.array([0.0, -self.gravity_mps2], dtype=float)
        total_accel = accel_cmd_mps2 + drag_accel_vec + thrust_accel_vec + gravity_vec
        return vel.copy(), total_accel

    def step(
        self,
        state: MissileState,
        accel_cmd_mps2: np.ndarray,
        dt_s: float,
        hold_cruise_speed: bool = False,
    ) -> MissileState:
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        if state.position_m.shape != (2,):
            raise ValueError(
                f"position_m must have shape (2,), got {state.position_m.shape}"
            )
        if state.velocity_mps.shape != (2,):
            raise ValueError(
                f"velocity_mps must have shape (2,), got {state.velocity_mps.shape}"
            )

        k1_pos, k1_vel = self._compute_derivatives(
            state, accel_cmd_mps2, hold_cruise_speed
        )

        s2 = MissileState(
            position_m=state.position_m + 0.5 * dt_s * k1_pos,
            velocity_mps=state.velocity_mps + 0.5 * dt_s * k1_vel,
        )
        k2_pos, k2_vel = self._compute_derivatives(
            s2, accel_cmd_mps2, hold_cruise_speed
        )

        s3 = MissileState(
            position_m=state.position_m + 0.5 * dt_s * k2_pos,
            velocity_mps=state.velocity_mps + 0.5 * dt_s * k2_vel,
        )
        k3_pos, k3_vel = self._compute_derivatives(
            s3, accel_cmd_mps2, hold_cruise_speed
        )

        s4 = MissileState(
            position_m=state.position_m + dt_s * k3_pos,
            velocity_mps=state.velocity_mps + dt_s * k3_vel,
        )
        k4_pos, k4_vel = self._compute_derivatives(
            s4, accel_cmd_mps2, hold_cruise_speed
        )

        p_next = state.position_m + (dt_s / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos)
        v_next = state.velocity_mps + (dt_s / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel)
        return MissileState(position_m=p_next, velocity_mps=v_next)

    def get_rcs_dbsm(
        self,
        missile_pos_m: np.ndarray,
        target_pos_m: np.ndarray,
        velocity_mps: np.ndarray,
    ) -> float:
        """Estimate RCS [dBsm] from simple aspect-angle model."""
        m_pos = np.asarray(missile_pos_m, dtype=float).reshape(2)
        t_pos = np.asarray(target_pos_m, dtype=float).reshape(2)
        vel = np.asarray(velocity_mps, dtype=float).reshape(2)

        los_vec = t_pos - m_pos
        los_norm = float(np.linalg.norm(los_vec))
        if los_norm < 1e-6:
            return 0.0

        vel_norm = float(np.linalg.norm(vel))
        if vel_norm < 1e-6:
            return 0.0

        los_hat = los_vec / los_norm
        vel_hat = vel / vel_norm
        cos_aspect = float(np.dot(los_hat, vel_hat))
        aspect_deg = float(np.degrees(np.arccos(np.clip(cos_aspect, -1.0, 1.0))))

        rcs_head_dbsm = 15.0
        rcs_tail_dbsm = 9.0
        aspect_norm = (aspect_deg - 90.0) / 90.0
        rcs_dbsm = rcs_head_dbsm - (rcs_head_dbsm - rcs_tail_dbsm) * (aspect_norm**2)
        return float(rcs_dbsm)

