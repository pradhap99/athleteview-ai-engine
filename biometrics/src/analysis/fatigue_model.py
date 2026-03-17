"""
LSTM-based fatigue prediction model.

Consumes a sliding window of biometric features (HR, HRV, SpO2, temperature,
activity level) and produces a fatigue score in the range [0, 100].
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class FatigueLSTM(nn.Module):
        """Two-layer LSTM regressor for fatigue estimation.

        Input features (per time-step):
            0 - heart_rate (normalised to [0, 1] via max_hr)
            1 - hrv_rmssd (normalised by resting baseline)
            2 - spo2 (normalised, 0-1)
            3 - core_temperature (normalised around 37 C)
            4 - activity_level (0-1)

        Output:
            Single scalar: fatigue score [0, 100].
        """

        def __init__(
            self,
            input_size: int = 5,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),  # Output in [0, 1]; scaled to [0, 100]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            x : Tensor, shape (batch, seq_len, input_size)

            Returns
            -------
            Tensor, shape (batch, 1)
                Fatigue score in [0, 1].
            """
            lstm_out, _ = self.lstm(x)
            # Use last hidden state
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden)

else:
    # Stub so the module can be imported without torch
    class FatigueLSTM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for FatigueLSTM")


# ---------------------------------------------------------------------------
# Fatigue model wrapper (inference)
# ---------------------------------------------------------------------------


class FatigueModel:
    """High-level wrapper for fatigue inference.

    Manages feature normalisation, sliding-window buffering, and model
    inference.

    Parameters
    ----------
    model_path : str or None
        Path to serialised FatigueLSTM state_dict.  If *None* or the file
        does not exist, a default (untrained) model is instantiated.
    input_size : int
        Number of per-step features.
    hidden_size : int
        LSTM hidden dimension.
    num_layers : int
        LSTM layer count.
    sequence_length : int
        Sliding-window length in time-steps.
    max_hr : float
        Max HR for normalisation.
    resting_hrv_rmssd : float
        Baseline HRV RMSSD (ms) for normalisation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 60,
        max_hr: float = 200.0,
        resting_hrv_rmssd: float = 50.0,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.max_hr = max_hr
        self.resting_hrv_rmssd = resting_hrv_rmssd

        # Feature buffer (ring)
        self._buffer: List[List[float]] = []

        # Load or create model
        self._model: Optional[FatigueLSTM] = None
        self._use_heuristic = True

        if TORCH_AVAILABLE:
            try:
                model = FatigueLSTM(input_size, hidden_size, num_layers)
                if model_path:
                    try:
                        state = torch.load(model_path, map_location="cpu", weights_only=True)
                        model.load_state_dict(state)
                    except (FileNotFoundError, RuntimeError):
                        pass  # Use untrained model
                model.eval()
                self._model = model
                self._use_heuristic = False
            except Exception:
                self._use_heuristic = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        heart_rate: float,
        hrv_rmssd: float,
        spo2: float,
        core_temp: float,
        activity_level: float,
    ) -> float:
        """Append a new time-step and return the current fatigue score.

        Parameters
        ----------
        heart_rate : float
            Current heart rate (bpm).
        hrv_rmssd : float
            Current HRV RMSSD (ms).
        spo2 : float
            SpO2 percentage (0-100).
        core_temp : float
            Estimated core temperature (Celsius).
        activity_level : float
            Normalised activity level (0-1).

        Returns
        -------
        float
            Fatigue score (0-100).
        """
        features = self._normalise(heart_rate, hrv_rmssd, spo2, core_temp, activity_level)
        self._buffer.append(features)

        # Keep only the last `sequence_length` entries
        if len(self._buffer) > self.sequence_length:
            self._buffer = self._buffer[-self.sequence_length:]

        if self._use_heuristic or self._model is None:
            return self._heuristic_fatigue(heart_rate, hrv_rmssd, spo2, core_temp, activity_level)

        return self._infer()

    def predict_batch(self, feature_matrix: np.ndarray) -> float:
        """Run inference on a pre-formed feature matrix.

        Parameters
        ----------
        feature_matrix : ndarray, shape (seq_len, input_size)
            Pre-normalised feature matrix.

        Returns
        -------
        float
            Fatigue score (0-100).
        """
        if self._use_heuristic or self._model is None:
            return 50.0

        x = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = self._model(x).item() * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def reset(self) -> None:
        """Clear the feature buffer."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalise(
        self,
        hr: float,
        hrv_rmssd: float,
        spo2: float,
        core_temp: float,
        activity: float,
    ) -> List[float]:
        """Normalise raw features to [0, 1] range."""
        hr_norm = hr / self.max_hr if self.max_hr > 0 else 0.0
        hrv_norm = hrv_rmssd / (self.resting_hrv_rmssd * 2.0) if self.resting_hrv_rmssd > 0 else 0.0
        spo2_norm = spo2 / 100.0
        temp_norm = (core_temp - 35.0) / 5.0  # Map 35-40 C to 0-1
        act_norm = float(np.clip(activity, 0.0, 1.0))

        return [
            float(np.clip(hr_norm, 0.0, 1.0)),
            float(np.clip(hrv_norm, 0.0, 1.0)),
            float(np.clip(spo2_norm, 0.0, 1.0)),
            float(np.clip(temp_norm, 0.0, 1.0)),
            act_norm,
        ]

    def _infer(self) -> float:
        """Run LSTM inference on the current buffer."""
        if len(self._buffer) < 2:
            return 0.0

        # Pad if buffer shorter than sequence_length
        buf = list(self._buffer)
        while len(buf) < self.sequence_length:
            buf.insert(0, buf[0])  # Repeat first entry

        arr = np.array(buf[-self.sequence_length:], dtype=np.float32)
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            score = self._model(x).item() * 100.0  # type: ignore[union-attr]

        return float(np.clip(score, 0.0, 100.0))

    def _heuristic_fatigue(
        self,
        hr: float,
        hrv_rmssd: float,
        spo2: float,
        core_temp: float,
        activity_level: float,
    ) -> float:
        """Rule-based fatigue estimation when LSTM is unavailable.

        Multi-factor weighted score:
        - HR elevation (30%): Higher HR relative to max -> more fatigued
        - HRV depression (25%): Lower HRV relative to baseline -> more fatigued
        - SpO2 decline (15%): Lower SpO2 -> more fatigued
        - Temperature elevation (15%): Higher core temp -> more fatigued
        - Sustained high activity (15%): Duration at high intensity
        """
        # HR component: (HR / max_HR)
        hr_factor = (hr / self.max_hr) if self.max_hr > 0 else 0.5

        # HRV depression: lower HRV = higher fatigue
        if self.resting_hrv_rmssd > 0:
            hrv_ratio = hrv_rmssd / self.resting_hrv_rmssd
            hrv_factor = max(0.0, 1.0 - hrv_ratio)  # Inverted
        else:
            hrv_factor = 0.5

        # SpO2: deviation from 100%
        spo2_factor = max(0.0, (100.0 - spo2) / 15.0)  # 85% -> ~1.0

        # Temperature: deviation from 37 C
        temp_factor = max(0.0, (core_temp - 37.0) / 3.0)  # 40C -> 1.0

        # Activity duration factor (from buffer length at high activity)
        high_activity_count = sum(
            1 for f in self._buffer if len(f) > 4 and f[4] > 0.6
        )
        duration_factor = min(high_activity_count / max(self.sequence_length, 1), 1.0)

        score = (
            0.30 * hr_factor
            + 0.25 * hrv_factor
            + 0.15 * spo2_factor
            + 0.15 * temp_factor
            + 0.15 * duration_factor
        ) * 100.0

        return float(np.clip(score, 0.0, 100.0))
