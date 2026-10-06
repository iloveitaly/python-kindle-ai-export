"""Tests for human-like pacing: jitter bounds, rolling-window budgets, breaks, and env parsing."""

from unittest.mock import AsyncMock

import pytest

from python_kindle_ai_export.pacing import HumanPacer, PacingConfig


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_pacer(
    config: PacingConfig,
    *,
    rand: float = 0.0,
    should_abort=lambda: False,
) -> tuple[HumanPacer, AsyncMock, FakeClock]:
    clock = FakeClock()

    async def fake_sleep(seconds: float) -> None:
        clock.now += seconds

    sleep = AsyncMock(side_effect=fake_sleep)
    pacer = HumanPacer(config, sleep_fn=sleep, clock_fn=clock, random_fn=lambda: rand, should_abort=should_abort)
    return pacer, sleep, clock


@pytest.mark.asyncio
@pytest.mark.parametrize(("rand", "expected"), [(0.0, 0.8), (1.0, 3.0), (0.5, 1.9)])
async def test_action_jitter_within_bounds(rand: float, expected: float) -> None:
    pacer, sleep, _ = make_pacer(PacingConfig(), rand=rand)
    await pacer.pause("click")
    sleep.assert_awaited_once()
    assert sleep.await_args is not None
    assert sleep.await_args.args[0] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_break_fires_when_cap_reached_and_resets_window() -> None:
    config = PacingConfig(
        action_delay=(1.0, 1.0),
        window_seconds=300,
        max_actions_per_window=(3, 3),
        break_seconds=(10.0, 10.0),
    )
    pacer, sleep, _ = make_pacer(config)

    for _ in range(3):
        await pacer.pause("turn")
    assert pacer.breaks_taken == 0

    await pacer.pause("turn")
    assert pacer.breaks_taken == 1
    # 3 jitters + 10 one-second break slices + 1 jitter
    assert sleep.await_count == 3 + 10 + 1
    assert pacer.total_paused_seconds == pytest.approx(14.0)

    # window was cleared: two more actions don't trigger another break
    await pacer.pause("turn")
    await pacer.pause("turn")
    assert pacer.breaks_taken == 1


@pytest.mark.asyncio
async def test_actions_outside_window_expire() -> None:
    config = PacingConfig(
        action_delay=(0.0, 0.0),
        window_seconds=60,
        max_actions_per_window=(2, 2),
        break_seconds=(30.0, 30.0),
    )
    pacer, _, clock = make_pacer(config)

    for _ in range(10):
        await pacer.pause("turn")
        clock.now += 31  # at most 2 actions in any 60s span
    assert pacer.breaks_taken == 0


@pytest.mark.asyncio
async def test_cap_is_rerolled_after_break() -> None:
    config = PacingConfig(
        action_delay=(0.0, 0.0),
        max_actions_per_window=(5, 9),
        break_seconds=(1.0, 1.0),
    )
    rand = {"value": 0.0}  # initial cap -> 5
    clock = FakeClock()
    pacer = HumanPacer(config, sleep_fn=AsyncMock(), clock_fn=clock, random_fn=lambda: rand["value"])
    assert pacer.current_cap == 5

    for _ in range(5):
        await pacer.pause("turn")
    rand["value"] = 0.99  # re-rolled cap -> 9
    await pacer.pause("turn")
    assert pacer.breaks_taken == 1
    assert pacer.current_cap == 9


@pytest.mark.asyncio
async def test_break_aborts_when_cancelled() -> None:
    state = {"cancelled": False}
    config = PacingConfig(
        action_delay=(0.0, 0.0),
        max_actions_per_window=(1, 1),
        break_seconds=(200.0, 200.0),
    )
    pacer, sleep, _ = make_pacer(config, should_abort=lambda: state["cancelled"])

    async def cancel_after_two(seconds: float) -> None:
        if sleep.await_count >= 2:
            state["cancelled"] = True

    sleep.side_effect = cancel_after_two
    await pacer.pause("turn")
    await pacer.pause("turn")
    assert pacer.breaks_taken == 1
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_disabled_never_sleeps() -> None:
    pacer, sleep, _ = make_pacer(PacingConfig(enabled=False, max_actions_per_window=(1, 1)))
    for _ in range(20):
        await pacer.pause("turn")
    sleep.assert_not_awaited()
    assert pacer.breaks_taken == 0


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EXTRACT_PACING",
        "EXTRACT_ACTION_DELAY",
        "EXTRACT_WINDOW_SECONDS",
        "EXTRACT_MAX_ACTIONS_PER_WINDOW",
        "EXTRACT_BREAK_SECONDS",
    ):
        for prefix in ("", "KINDLE_EXPORT_", "KINDLE_AI_EXPORT_"):
            monkeypatch.delenv(f"{prefix}{name}", raising=False)
    assert PacingConfig.from_env() == PacingConfig()


def test_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACT_PACING", "off")
    monkeypatch.setenv("EXTRACT_ACTION_DELAY", "1-2.5")
    monkeypatch.setenv("EXTRACT_WINDOW_SECONDS", "120")
    monkeypatch.setenv("EXTRACT_MAX_ACTIONS_PER_WINDOW", "40")
    monkeypatch.setenv("EXTRACT_BREAK_SECONDS", "60-90")
    config = PacingConfig.from_env()
    assert config.enabled is False
    assert config.action_delay == (1.0, 2.5)
    assert config.window_seconds == 120.0
    assert config.max_actions_per_window == (40, 40)
    assert config.break_seconds == (60.0, 90.0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EXTRACT_ACTION_DELAY", "3-1"),
        ("EXTRACT_ACTION_DELAY", "abc"),
        ("EXTRACT_BREAK_SECONDS", "1-2-3"),
        ("EXTRACT_WINDOW_SECONDS", "0"),
        ("EXTRACT_MAX_ACTIONS_PER_WINDOW", "0"),
        ("EXTRACT_PACING", "maybe"),
    ],
)
def test_from_env_rejects_invalid(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        PacingConfig.from_env()
