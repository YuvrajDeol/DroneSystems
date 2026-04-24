from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Centralized high-level simulation constants."""

    # Missile Flight Parameters
    CRUISE_ALTITUDE_RANGE_M: tuple[float, float] = (
        4.5,
        60.0,
    )  # 15 to 200 ft. Lower evades radar; higher increases ditch risk.
    CRUISE_SPEED_MACH_RANGE: tuple[float, float] = (
        0.8,
        3.5,
    )  # Dictates closing velocity and available terminal energy.
    LATERAL_G_LIMIT: float = 15.0  # Max horizontal turning capability.
    VERTICAL_G_LIMIT: float = 7.0  # Max pitch-plane acceleration before stall/structural risk.
    GUIDANCE_GAIN_N: float = 4.0  # PN gain N' (typical range 2 to 5).
    TERMINAL_RANGE_TRIGGER_M: float = (
        15000.0
    )  # Trigger distance for popup+dive geometry timing.

    # Sensor & Noise Parameters
    RADAR_ALTIMETER_BIAS_M: float = 2.0  # Constant sensor bias that causes altitude drift.
    ALT_DEVIATION_TOLERANCE_M: float = 0.3  # Max acceptable altitude-tracking variance.

    # Environmental & Target Parameters
    SEA_STATE_RANGE: tuple[int, int] = (
        0,
        8,
    )  # Beaufort-like wave state affecting aero/radar multipath.
    TARGET_RADAR_HEIGHT_M: tuple[float, float] = (
        10.0,
        40.0,
    )  # Target radar mast height drives horizon geometry.
    EVAPORATION_DUCT_HEIGHT_M: tuple[float, float] = (
        0.0,
        40.0,
    )  # Traps microwave energy and changes detectability.
    RADAR_FREQ_GHZ_RANGE: tuple[float, float] = (
        10.0,
        12.0,
    )  # X-band operation affects reflection and phase behavior.
    AIR_TEMP_C_RANGE: tuple[float, float] = (
        -20.0,
        50.0,
    )  # Affects refractivity, speed of sound, and aero coefficients.
