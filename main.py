from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.config import SimulationConfig
from src.guidance.guidance import SeaSkimGuidance
from src.output.data_logger import DataLogger
from src.physics.missile import Missile, MissileState


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _export_sim_log_txt(rows: list[dict[str, object]], root: Path) -> Path:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = logs_dir / f"sim_log_{timestamp}.txt"

    with txt_path.open("w", encoding="utf-8", newline="") as f:
        f.write("Simulation Log Export\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Rows: {len(rows)}\n\n")

        if not rows:
            f.write("No simulation rows were recorded.\n")
            return txt_path

        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return txt_path


def main() -> int:
    root = Path(__file__).resolve().parent
    sim_constants = SimulationConfig()
    config_paths = [
        root / "config" / "sim_config.yaml",
        root / "sim_config.yaml",
    ]
    config_path: Path | None = None
    for candidate in config_paths:
        if candidate.exists():
            config_path = candidate
            break
    if config_path is None:
        raise FileNotFoundError(
            f"Could not find sim_config.yaml in {config_paths[0]} or {config_paths[1]}"
        )
    cfg = _load_config(config_path)

    dt_s = float(cfg["sim"]["dt_s"])
    duration_s = float(cfg["sim"]["duration_s"])
    steps = int(np.ceil(duration_s / dt_s))

    missile = Missile(
        mass_kg=float(cfg["missile"]["mass_kg"]),
        max_accel_mps2=float(cfg["missile"]["max_accel_mps2"]),
        air_density_kgpm3=float(cfg["missile"]["air_density_kgpm3"]),
        drag_area_m2=float(cfg["missile"]["drag_area_m2"]),
        cruise_speed_mps=float(cfg["missile"]["cruise_speed_mps"]),
        speed_hold_kp=float(cfg["missile"]["speed_hold_kp"]),
        max_thrust_n=float(cfg["missile"]["max_thrust_n"]),
    )
    guidance = SeaSkimGuidance(
        cruise_alt_m=float(cfg["guidance"]["cruise_alt_m"]),
        terminal_range_m=max(
            float(cfg["guidance"]["terminal_range_m"]),
            sim_constants.TERMINAL_RANGE_TRIGGER_M,
        ),
        popup_alt_m=float(cfg["guidance"]["popup_alt_m"]),
        max_accel_mps2=float(cfg["guidance"]["max_accel_mps2"]),
        max_vz_cmd_mps=float(cfg["guidance"]["max_vz_cmd_mps"]),
        alt_to_vz_gain=float(cfg["guidance"]["alt_to_vz_gain"]),
        vz_kp=float(cfg["guidance"]["vz_kp"]),
        vz_ki=float(cfg["guidance"]["vz_ki"]),
        vz_kd=float(cfg["guidance"]["vz_kd"]),
        N_gain=float(cfg["guidance"].get("N_gain", sim_constants.GUIDANCE_GAIN_N)),
    )

    m_state = MissileState(
        position_m=np.array(cfg["missile"]["init_position_m"], dtype=float),
        velocity_mps=np.array(cfg["missile"]["init_velocity_mps"], dtype=float),
    )
    t_pos = np.array(cfg["target"]["init_position_m"], dtype=float)
    t_vel = np.array(cfg["target"]["init_velocity_mps"], dtype=float)
    speed_of_sound_mps = float(cfg["sim"]["speed_of_sound_mps"])

    logger = DataLogger(out_dir=(root / cfg["output"]["out_dir"]).resolve())

    for k in range(steps):
        t_s = k * dt_s

        a_cmd = guidance.accel_command(
            missile_pos_m=m_state.position_m,
            missile_vel_mps=m_state.velocity_mps,
            target_pos_m=t_pos,
            target_vel_mps=t_vel,
            dt_s=dt_s,
        )

        # Disable speed-hold in terminal dive so drag can bleed kinetic energy.
        hold_cruise_speed = guidance.phase in {"cruise", "popup"}
        m_state = missile.step(m_state, a_cmd, dt_s, hold_cruise_speed=hold_cruise_speed)
        # Keep sea level at z=0 and altitude AGL non-negative.
        position_m = m_state.position_m.copy()
        velocity_mps = m_state.velocity_mps.copy()
        if position_m[1] <= 0.0:
            position_m[1] = 0.0
            velocity_mps[1] = max(0.0, velocity_mps[1])
        m_state = MissileState(
            position_m=position_m,
            velocity_mps=velocity_mps,
        )
        t_pos = t_pos + t_vel * dt_s

        if m_state.position_m[1] <= 0.0:
            miss_distance_m = abs(float(t_pos[0] - m_state.position_m[0]))
            mach = float(np.linalg.norm(m_state.velocity_mps) / speed_of_sound_mps)
            rcs_dbsm = missile.get_rcs_dbsm(
                missile_pos_m=m_state.position_m,
                target_pos_m=t_pos,
                velocity_mps=m_state.velocity_mps,
            )
            logger.log(
                t_s=t_s,
                x_m=m_state.position_m[0],
                z_m=m_state.position_m[1],
                vx=m_state.velocity_mps[0],
                vz=m_state.velocity_mps[1],
                mach=mach,
                miss_distance_m=miss_distance_m,
                rcs_dbsm=rcs_dbsm,
                guidance_phase=guidance.phase,
            )
            print(f"Impact! Simulation terminated at t={t_s:.2f}s")
            print(f"Final Miss Distance: {miss_distance_m:.2f} meters")
            break

        miss_distance_m = float(np.linalg.norm(t_pos - m_state.position_m))
        mach = float(np.linalg.norm(m_state.velocity_mps) / speed_of_sound_mps)
        rcs_dbsm = missile.get_rcs_dbsm(
            missile_pos_m=m_state.position_m,
            target_pos_m=t_pos,
            velocity_mps=m_state.velocity_mps,
        )

        logger.log(
            t_s=t_s,
            x_m=m_state.position_m[0],
            z_m=m_state.position_m[1],
            vx=m_state.velocity_mps[0],
            vz=m_state.velocity_mps[1],
            mach=mach,
            miss_distance_m=miss_distance_m,
            rcs_dbsm=rcs_dbsm,
            guidance_phase=guidance.phase,
        )

        if miss_distance_m < 5.0:
            break

    out_path = logger.to_csv(filename=str(cfg["output"]["csv_name"]))
    txt_out_path = _export_sim_log_txt(logger._rows, root)
    print(f"Wrote {out_path}")
    print(f"Simulation text log saved to {txt_out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

