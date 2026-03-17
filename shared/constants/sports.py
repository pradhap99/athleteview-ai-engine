"""
AthleteView AI Platform - Sport-specific constants.

Defines enumerations, field dimensions, player counts, positions, highlight events,
biometric zones, overlay positions, and color schemes for each supported sport.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Sport type enumeration
# ---------------------------------------------------------------------------

class SportType(str, Enum):
    """Supported sports on the AthleteView platform."""
    CRICKET = "cricket"
    FOOTBALL = "football"
    KABADDI = "kabaddi"
    BASKETBALL = "basketball"
    TENNIS = "tennis"
    HOCKEY = "hockey"
    BADMINTON = "badminton"
    SWIMMING = "swimming"


# ---------------------------------------------------------------------------
# Field / court / arena dimensions (in metres)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldDimensions:
    """Physical dimensions of a playing area."""
    length: float
    width: float
    unit: str = "metres"
    notes: str = ""


FIELD_DIMENSIONS: Dict[SportType, FieldDimensions] = {
    SportType.CRICKET: FieldDimensions(
        length=150.0, width=150.0,
        notes="Oval ground; pitch is 20.12m long",
    ),
    SportType.FOOTBALL: FieldDimensions(
        length=105.0, width=68.0,
        notes="FIFA standard pitch",
    ),
    SportType.KABADDI: FieldDimensions(
        length=13.0, width=10.0,
        notes="Pro Kabaddi standard mat",
    ),
    SportType.BASKETBALL: FieldDimensions(
        length=28.0, width=15.0,
        notes="FIBA standard court",
    ),
    SportType.TENNIS: FieldDimensions(
        length=23.77, width=10.97,
        notes="ITF standard doubles court",
    ),
    SportType.HOCKEY: FieldDimensions(
        length=91.4, width=55.0,
        notes="FIH standard field",
    ),
    SportType.BADMINTON: FieldDimensions(
        length=13.4, width=6.1,
        notes="BWF standard doubles court",
    ),
    SportType.SWIMMING: FieldDimensions(
        length=50.0, width=25.0,
        notes="Olympic standard 50m pool",
    ),
}


# ---------------------------------------------------------------------------
# Player counts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlayerCount:
    """Number of players per team for a sport."""
    per_team: int
    substitutes: int
    total_on_field: int  # both teams combined


PLAYER_COUNTS: Dict[SportType, PlayerCount] = {
    SportType.CRICKET: PlayerCount(per_team=11, substitutes=4, total_on_field=22),
    SportType.FOOTBALL: PlayerCount(per_team=11, substitutes=7, total_on_field=22),
    SportType.KABADDI: PlayerCount(per_team=7, substitutes=5, total_on_field=14),
    SportType.BASKETBALL: PlayerCount(per_team=5, substitutes=7, total_on_field=10),
    SportType.TENNIS: PlayerCount(per_team=1, substitutes=0, total_on_field=2),
    SportType.HOCKEY: PlayerCount(per_team=11, substitutes=5, total_on_field=22),
    SportType.BADMINTON: PlayerCount(per_team=1, substitutes=0, total_on_field=2),
    SportType.SWIMMING: PlayerCount(per_team=1, substitutes=0, total_on_field=8),
}


# ---------------------------------------------------------------------------
# Positions per sport
# ---------------------------------------------------------------------------

POSITIONS: Dict[SportType, List[str]] = {
    SportType.CRICKET: [
        "batsman", "bowler", "all_rounder", "wicket_keeper",
        "opening_batsman", "middle_order", "tail_ender",
        "spin_bowler", "fast_bowler", "captain",
    ],
    SportType.FOOTBALL: [
        "goalkeeper", "centre_back", "full_back", "wing_back",
        "defensive_midfielder", "central_midfielder", "attacking_midfielder",
        "winger", "striker", "forward",
    ],
    SportType.KABADDI: [
        "raider", "left_corner", "right_corner",
        "left_cover", "right_cover", "left_in", "right_in",
    ],
    SportType.BASKETBALL: [
        "point_guard", "shooting_guard", "small_forward",
        "power_forward", "center",
    ],
    SportType.TENNIS: [
        "singles_player", "doubles_player",
    ],
    SportType.HOCKEY: [
        "goalkeeper", "full_back", "half_back",
        "centre_half", "inner", "wing", "centre_forward",
    ],
    SportType.BADMINTON: [
        "singles_player", "doubles_player",
    ],
    SportType.SWIMMING: [
        "freestyle", "backstroke", "breaststroke",
        "butterfly", "individual_medley",
    ],
}


# ---------------------------------------------------------------------------
# Highlight events per sport
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HighlightDef:
    """Definition of a highlight event within a sport."""
    key: str
    label: str
    default_clip_seconds: int = 10
    priority: int = 1  # 1 = highest


HIGHLIGHT_EVENTS: Dict[SportType, List[HighlightDef]] = {
    SportType.CRICKET: [
        HighlightDef(key="wicket", label="Wicket", default_clip_seconds=15, priority=1),
        HighlightDef(key="boundary", label="Boundary (4)", default_clip_seconds=10, priority=2),
        HighlightDef(key="six", label="Six", default_clip_seconds=12, priority=1),
        HighlightDef(key="catch", label="Catch", default_clip_seconds=10, priority=2),
        HighlightDef(key="runout", label="Run Out", default_clip_seconds=12, priority=2),
        HighlightDef(key="lbw", label="LBW", default_clip_seconds=10, priority=2),
        HighlightDef(key="no_ball", label="No Ball", default_clip_seconds=8, priority=3),
    ],
    SportType.FOOTBALL: [
        HighlightDef(key="goal", label="Goal", default_clip_seconds=15, priority=1),
        HighlightDef(key="foul", label="Foul", default_clip_seconds=8, priority=3),
        HighlightDef(key="corner", label="Corner Kick", default_clip_seconds=10, priority=3),
        HighlightDef(key="penalty", label="Penalty", default_clip_seconds=20, priority=1),
        HighlightDef(key="red_card", label="Red Card", default_clip_seconds=12, priority=1),
        HighlightDef(key="yellow_card", label="Yellow Card", default_clip_seconds=8, priority=2),
        HighlightDef(key="offside", label="Offside", default_clip_seconds=8, priority=3),
    ],
    SportType.KABADDI: [
        HighlightDef(key="raid", label="Successful Raid", default_clip_seconds=10, priority=1),
        HighlightDef(key="tackle", label="Tackle", default_clip_seconds=10, priority=1),
        HighlightDef(key="bonus", label="Bonus Point", default_clip_seconds=8, priority=2),
        HighlightDef(key="super_raid", label="Super Raid", default_clip_seconds=12, priority=1),
        HighlightDef(key="all_out", label="All Out", default_clip_seconds=15, priority=1),
        HighlightDef(key="do_or_die", label="Do or Die Raid", default_clip_seconds=10, priority=2),
    ],
    SportType.BASKETBALL: [
        HighlightDef(key="dunk", label="Dunk", default_clip_seconds=8, priority=1),
        HighlightDef(key="three", label="Three-Pointer", default_clip_seconds=8, priority=1),
        HighlightDef(key="block", label="Block", default_clip_seconds=8, priority=2),
        HighlightDef(key="steal", label="Steal", default_clip_seconds=8, priority=2),
        HighlightDef(key="alley_oop", label="Alley-Oop", default_clip_seconds=10, priority=1),
        HighlightDef(key="fast_break", label="Fast Break", default_clip_seconds=10, priority=2),
    ],
    SportType.TENNIS: [
        HighlightDef(key="ace", label="Ace", default_clip_seconds=6, priority=1),
        HighlightDef(key="break_point", label="Break Point", default_clip_seconds=10, priority=1),
        HighlightDef(key="match_point", label="Match Point", default_clip_seconds=15, priority=1),
        HighlightDef(key="rally", label="Long Rally", default_clip_seconds=20, priority=2),
    ],
    SportType.HOCKEY: [
        HighlightDef(key="goal", label="Goal", default_clip_seconds=12, priority=1),
        HighlightDef(key="penalty_corner", label="Penalty Corner", default_clip_seconds=15, priority=1),
        HighlightDef(key="save", label="Goalkeeper Save", default_clip_seconds=8, priority=2),
        HighlightDef(key="card", label="Card", default_clip_seconds=8, priority=2),
    ],
    SportType.BADMINTON: [
        HighlightDef(key="smash", label="Smash", default_clip_seconds=6, priority=1),
        HighlightDef(key="rally", label="Long Rally", default_clip_seconds=20, priority=2),
        HighlightDef(key="net_play", label="Net Play", default_clip_seconds=8, priority=2),
    ],
    SportType.SWIMMING: [
        HighlightDef(key="record", label="Record Broken", default_clip_seconds=15, priority=1),
        HighlightDef(key="photo_finish", label="Photo Finish", default_clip_seconds=12, priority=1),
        HighlightDef(key="turn", label="Underwater Turn", default_clip_seconds=8, priority=3),
    ],
}


# ---------------------------------------------------------------------------
# Biometric zones per sport
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeartRateZone:
    """A named heart-rate zone expressed as percentage of max HR."""
    name: str
    min_pct: float  # percentage of max HR
    max_pct: float


@dataclass(frozen=True)
class BiometricZoneConfig:
    """Sport-specific biometric thresholds and zones."""
    hr_zones: List[HeartRateZone]
    max_safe_hr_pct: float  # percentage of max HR before alert
    effort_low_threshold: float  # fatigue score below this = low effort
    effort_high_threshold: float  # fatigue score above this = high effort
    spo2_alert_threshold: float  # SpO2 below this triggers alert


_STANDARD_HR_ZONES: List[HeartRateZone] = [
    HeartRateZone(name="recovery", min_pct=0.50, max_pct=0.60),
    HeartRateZone(name="aerobic", min_pct=0.60, max_pct=0.70),
    HeartRateZone(name="tempo", min_pct=0.70, max_pct=0.80),
    HeartRateZone(name="threshold", min_pct=0.80, max_pct=0.90),
    HeartRateZone(name="anaerobic", min_pct=0.90, max_pct=1.00),
]


BIOMETRIC_ZONES: Dict[SportType, BiometricZoneConfig] = {
    SportType.CRICKET: BiometricZoneConfig(
        hr_zones=_STANDARD_HR_ZONES,
        max_safe_hr_pct=0.92,
        effort_low_threshold=0.25,
        effort_high_threshold=0.70,
        spo2_alert_threshold=93.0,
    ),
    SportType.FOOTBALL: BiometricZoneConfig(
        hr_zones=_STANDARD_HR_ZONES,
        max_safe_hr_pct=0.95,
        effort_low_threshold=0.20,
        effort_high_threshold=0.75,
        spo2_alert_threshold=92.0,
    ),
    SportType.KABADDI: BiometricZoneConfig(
        hr_zones=_STANDARD_HR_ZONES,
        max_safe_hr_pct=0.95,
        effort_low_threshold=0.20,
        effort_high_threshold=0.80,
        spo2_alert_threshold=92.0,
    ),
    SportType.BASKETBALL: BiometricZoneConfig(
        hr_zones=_STANDARD_HR_ZONES,
        max_safe_hr_pct=0.95,
        effort_low_threshold=0.20,
        effort_high_threshold=0.78,
        spo2_alert_threshold=92.0,
    ),
    SportType.TENNIS: BiometricZoneConfig(
        hr_zones=_STANDARD_HR_ZONES,
        max_safe_hr_pct=0.93,
        effort_low_threshold=0.22,
        effort_high_threshold=0.72,
        spo2_alert_threshold=93.0,
    ),
    SportType.HOCKEY: BiometricZoneConfig(
        hr_zones=_STANDARD_HR_ZONES,
        max_safe_hr_pct=0.95,
        effort_low_threshold=0.20,
        effort_high_threshold=0.76,
        spo2_alert_threshold=92.0,
    ),
    SportType.BADMINTON: BiometricZoneConfig(
        hr_zones=_STANDARD_HR_ZONES,
        max_safe_hr_pct=0.93,
        effort_low_threshold=0.22,
        effort_high_threshold=0.72,
        spo2_alert_threshold=93.0,
    ),
    SportType.SWIMMING: BiometricZoneConfig(
        hr_zones=_STANDARD_HR_ZONES,
        max_safe_hr_pct=0.94,
        effort_low_threshold=0.18,
        effort_high_threshold=0.80,
        spo2_alert_threshold=91.0,
    ),
}


# ---------------------------------------------------------------------------
# Default overlay positions per sport
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OverlayPosition:
    """
    Position of a UI overlay element on the stream, expressed as percentage
    offsets from the top-left corner of the video frame.
    """
    x_pct: float
    y_pct: float
    width_pct: float
    height_pct: float


@dataclass(frozen=True)
class OverlayLayout:
    """Default overlay element positions for a sport."""
    scoreboard: OverlayPosition
    biometrics_panel: OverlayPosition
    tracking_info: OverlayPosition
    highlight_badge: OverlayPosition


OVERLAY_POSITIONS: Dict[SportType, OverlayLayout] = {
    SportType.CRICKET: OverlayLayout(
        scoreboard=OverlayPosition(x_pct=2.0, y_pct=2.0, width_pct=30.0, height_pct=8.0),
        biometrics_panel=OverlayPosition(x_pct=75.0, y_pct=2.0, width_pct=23.0, height_pct=25.0),
        tracking_info=OverlayPosition(x_pct=2.0, y_pct=88.0, width_pct=40.0, height_pct=10.0),
        highlight_badge=OverlayPosition(x_pct=40.0, y_pct=5.0, width_pct=20.0, height_pct=6.0),
    ),
    SportType.FOOTBALL: OverlayLayout(
        scoreboard=OverlayPosition(x_pct=35.0, y_pct=2.0, width_pct=30.0, height_pct=6.0),
        biometrics_panel=OverlayPosition(x_pct=75.0, y_pct=15.0, width_pct=23.0, height_pct=25.0),
        tracking_info=OverlayPosition(x_pct=2.0, y_pct=88.0, width_pct=35.0, height_pct=10.0),
        highlight_badge=OverlayPosition(x_pct=2.0, y_pct=2.0, width_pct=20.0, height_pct=6.0),
    ),
    SportType.KABADDI: OverlayLayout(
        scoreboard=OverlayPosition(x_pct=30.0, y_pct=2.0, width_pct=40.0, height_pct=8.0),
        biometrics_panel=OverlayPosition(x_pct=75.0, y_pct=15.0, width_pct=23.0, height_pct=30.0),
        tracking_info=OverlayPosition(x_pct=2.0, y_pct=85.0, width_pct=30.0, height_pct=13.0),
        highlight_badge=OverlayPosition(x_pct=2.0, y_pct=2.0, width_pct=22.0, height_pct=6.0),
    ),
    SportType.BASKETBALL: OverlayLayout(
        scoreboard=OverlayPosition(x_pct=30.0, y_pct=2.0, width_pct=40.0, height_pct=8.0),
        biometrics_panel=OverlayPosition(x_pct=75.0, y_pct=15.0, width_pct=23.0, height_pct=25.0),
        tracking_info=OverlayPosition(x_pct=2.0, y_pct=88.0, width_pct=35.0, height_pct=10.0),
        highlight_badge=OverlayPosition(x_pct=2.0, y_pct=2.0, width_pct=20.0, height_pct=6.0),
    ),
    SportType.TENNIS: OverlayLayout(
        scoreboard=OverlayPosition(x_pct=2.0, y_pct=2.0, width_pct=25.0, height_pct=15.0),
        biometrics_panel=OverlayPosition(x_pct=75.0, y_pct=2.0, width_pct=23.0, height_pct=20.0),
        tracking_info=OverlayPosition(x_pct=2.0, y_pct=88.0, width_pct=30.0, height_pct=10.0),
        highlight_badge=OverlayPosition(x_pct=35.0, y_pct=2.0, width_pct=20.0, height_pct=6.0),
    ),
    SportType.HOCKEY: OverlayLayout(
        scoreboard=OverlayPosition(x_pct=35.0, y_pct=2.0, width_pct=30.0, height_pct=6.0),
        biometrics_panel=OverlayPosition(x_pct=75.0, y_pct=15.0, width_pct=23.0, height_pct=25.0),
        tracking_info=OverlayPosition(x_pct=2.0, y_pct=88.0, width_pct=35.0, height_pct=10.0),
        highlight_badge=OverlayPosition(x_pct=2.0, y_pct=2.0, width_pct=20.0, height_pct=6.0),
    ),
    SportType.BADMINTON: OverlayLayout(
        scoreboard=OverlayPosition(x_pct=35.0, y_pct=2.0, width_pct=30.0, height_pct=8.0),
        biometrics_panel=OverlayPosition(x_pct=75.0, y_pct=2.0, width_pct=23.0, height_pct=20.0),
        tracking_info=OverlayPosition(x_pct=2.0, y_pct=88.0, width_pct=30.0, height_pct=10.0),
        highlight_badge=OverlayPosition(x_pct=2.0, y_pct=2.0, width_pct=20.0, height_pct=6.0),
    ),
    SportType.SWIMMING: OverlayLayout(
        scoreboard=OverlayPosition(x_pct=2.0, y_pct=2.0, width_pct=96.0, height_pct=10.0),
        biometrics_panel=OverlayPosition(x_pct=75.0, y_pct=15.0, width_pct=23.0, height_pct=25.0),
        tracking_info=OverlayPosition(x_pct=2.0, y_pct=85.0, width_pct=96.0, height_pct=13.0),
        highlight_badge=OverlayPosition(x_pct=40.0, y_pct=40.0, width_pct=20.0, height_pct=8.0),
    ),
}


# ---------------------------------------------------------------------------
# Color schemes per sport / team
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColorScheme:
    """RGBA hex color scheme for a team or sport's default branding."""
    primary: str
    secondary: str
    accent: str
    text: str
    background: str


# Default sport-level color schemes (used when team-specific colours are unavailable).
SPORT_COLOR_SCHEMES: Dict[SportType, ColorScheme] = {
    SportType.CRICKET: ColorScheme(
        primary="#1B5E20",
        secondary="#A5D6A7",
        accent="#FFD600",
        text="#FFFFFF",
        background="#0D3311",
    ),
    SportType.FOOTBALL: ColorScheme(
        primary="#1A237E",
        secondary="#7986CB",
        accent="#00E676",
        text="#FFFFFF",
        background="#0D1240",
    ),
    SportType.KABADDI: ColorScheme(
        primary="#E65100",
        secondary="#FFB74D",
        accent="#FF1744",
        text="#FFFFFF",
        background="#4E1A00",
    ),
    SportType.BASKETBALL: ColorScheme(
        primary="#B71C1C",
        secondary="#EF9A9A",
        accent="#FFD600",
        text="#FFFFFF",
        background="#3E0808",
    ),
    SportType.TENNIS: ColorScheme(
        primary="#004D40",
        secondary="#80CBC4",
        accent="#FFAB00",
        text="#FFFFFF",
        background="#00251A",
    ),
    SportType.HOCKEY: ColorScheme(
        primary="#0D47A1",
        secondary="#90CAF9",
        accent="#FF6D00",
        text="#FFFFFF",
        background="#062550",
    ),
    SportType.BADMINTON: ColorScheme(
        primary="#4A148C",
        secondary="#CE93D8",
        accent="#00E5FF",
        text="#FFFFFF",
        background="#1A0533",
    ),
    SportType.SWIMMING: ColorScheme(
        primary="#01579B",
        secondary="#81D4FA",
        accent="#FFFF00",
        text="#FFFFFF",
        background="#002F4F",
    ),
}

# Team-specific color schemes (keyed by lowercase team name).
TEAM_COLOR_SCHEMES: Dict[str, ColorScheme] = {
    "mumbai_indians": ColorScheme(
        primary="#004BA0",
        secondary="#B3D4FC",
        accent="#D4AF37",
        text="#FFFFFF",
        background="#002350",
    ),
    "chennai_super_kings": ColorScheme(
        primary="#F9CD05",
        secondary="#FFF176",
        accent="#0D47A1",
        text="#000000",
        background="#7B6503",
    ),
    "royal_challengers": ColorScheme(
        primary="#D32F2F",
        secondary="#EF9A9A",
        accent="#FFD600",
        text="#FFFFFF",
        background="#6B1818",
    ),
    "kolkata_knight_riders": ColorScheme(
        primary="#3A0078",
        secondary="#D1C4E9",
        accent="#FFD700",
        text="#FFFFFF",
        background="#1A003A",
    ),
    "bengaluru_fc": ColorScheme(
        primary="#003DA5",
        secondary="#6D9FFF",
        accent="#E53935",
        text="#FFFFFF",
        background="#001D52",
    ),
    "patna_pirates": ColorScheme(
        primary="#1B5E20",
        secondary="#A5D6A7",
        accent="#FF6F00",
        text="#FFFFFF",
        background="#0A3010",
    ),
}
