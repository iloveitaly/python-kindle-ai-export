"""Human-like pacing for browser actions: per-action jitter plus rolling-window action budgets."""

import random
import time
from collections import deque
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, model_validator

from .log import log
from .utils import get_env


class PacingConfig(BaseModel):
    """Tunable parameters for human-like browser pacing."""

    enabled: bool = True
    action_delay: tuple[float, float] = (0.8, 3.0)
    window_seconds: float = 300.0
    max_actions_per_window: tuple[int, int] = (50, 70)
    break_seconds: tuple[float, float] = (120.0, 240.0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> "PacingConfig":
        for name in ("action_delay", "max_actions_per_window", "break_seconds"):
            low, high = getattr(self, name)
            if low < 0 or high < low:
                raise ValueError(f"{name} must be a non-negative range with min <= max")
        if self.max_actions_per_window[0] < 1:
            raise ValueError("max_actions_per_window must be at least 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        return self

    @classmethod
    def from_env(cls) -> "PacingConfig":
        """Build config from EXTRACT_* environment variables, falling back to defaults."""
        defaults = cls()
        enabled_raw = (get_env("EXTRACT_PACING") or "on").strip().lower()
        if enabled_raw not in {"on", "off", "true", "false", "1", "0"}:
            raise ValueError("EXTRACT_PACING must be 'on' or 'off'")

        window_raw = get_env("EXTRACT_WINDOW_SECONDS")
        try:
            window_seconds = float(window_raw) if window_raw else defaults.window_seconds
        except ValueError as err:
            raise ValueError("EXTRACT_WINDOW_SECONDS must be a number") from err

        max_actions = _parse_range("EXTRACT_MAX_ACTIONS_PER_WINDOW", defaults.max_actions_per_window)
        return cls(
            enabled=enabled_raw in {"on", "true", "1"},
            action_delay=_parse_range("EXTRACT_ACTION_DELAY", defaults.action_delay),
            window_seconds=window_seconds,
            max_actions_per_window=(int(max_actions[0]), int(max_actions[1])),
            break_seconds=_parse_range("EXTRACT_BREAK_SECONDS", defaults.break_seconds),
        )


def _parse_range(name: str, default: tuple[float, float]) -> tuple[float, float]:
    """Parse a ``min-max`` (or single value) env var into a float range."""
    raw = get_env(name)
    if not raw:
        return (float(default[0]), float(default[1]))
    parts = [p.strip() for p in raw.split("-")]
    try:
        if len(parts) == 1:
            value = float(parts[0])
            return (value, value)
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    except ValueError:
        pass
    raise ValueError(f"{name} must be a number or a 'min-max' range")


class HumanPacer:
    """Paces visible browser actions to mimic a human reading session.

    Non-obvious behavior and design decisions:
    1. Rolling Window: Tracks action timestamps in a deque and enforces "no more than N actions
       in any ``window_seconds`` span", which avoids bursts slipping across fixed bucket edges.
    2. Randomized Budget: The per-window cap is re-rolled from ``max_actions_per_window`` after
       every break, so breaks do not land on a fixed, fingerprintable action count.
    3. Interruptible Breaks: Long breaks sleep in <=1s slices and poll ``should_abort`` so Ctrl+C
       cancellation is honored promptly instead of after several minutes.
    4. Test Injection: ``sleep_fn``, ``clock_fn``, and ``random_fn`` are injectable for
       deterministic unit tests without wall-clock waiting.
    """

    def __init__(
        self,
        config: PacingConfig,
        sleep_fn: Callable[[float], Awaitable[None]],
        clock_fn: Callable[[], float] = time.monotonic,
        random_fn: Callable[[], float] = random.random,
        should_abort: Callable[[], bool] = lambda: False,
    ) -> None:
        self.config = config
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._random = random_fn
        self._should_abort = should_abort
        self._actions: deque[float] = deque()
        self.current_cap = self._roll_cap()
        self.breaks_taken = 0
        self.total_paused_seconds = 0.0

    def _uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self._random()

    def _roll_cap(self) -> int:
        low, high = self.config.max_actions_per_window
        return min(high, low + int((high - low + 1) * self._random()))

    def _expire(self, now: float) -> None:
        cutoff = now - self.config.window_seconds
        while self._actions and self._actions[0] <= cutoff:
            self._actions.popleft()

    async def pause(self, action: str) -> None:
        """Wait an appropriate, human-like amount of time before a visible browser action."""
        if not self.config.enabled:
            return

        self._expire(self._clock())
        if len(self._actions) >= self.current_cap:
            await self._take_break(action)

        delay = self._uniform(*self.config.action_delay)
        if delay > 0:
            await self._sleep(delay)
            self.total_paused_seconds += delay
        self._actions.append(self._clock())

    async def _take_break(self, next_action: str) -> None:
        duration = self._uniform(*self.config.break_seconds)
        log.info(
            "taking reading break",
            seconds=round(duration, 1),
            actions_in_window=len(self._actions),
            next_action=next_action,
        )
        remaining = duration
        while remaining > 0:
            if self._should_abort():
                break
            step = min(1.0, remaining)
            await self._sleep(step)
            remaining -= step
            self.total_paused_seconds += step

        self._actions.clear()
        self.current_cap = self._roll_cap()
        self.breaks_taken += 1
