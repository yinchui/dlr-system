# DLR Safety Factor Publication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the static rating baseline and publish every steady and transient DLR value with one fixed `0.8` safety factor.

**Architecture:** Keep the IEEE 738 thermal adapter unchanged and apply the safety factor once inside `DlrPipeline.run` after all raw thermal calculations finish. Store only factored ratings in `DlrPipelineResult`, legacy/session data, transient results, downloads, and sag snapshots; keep the safety factor as backend metadata and remove static-baseline UI/helper paths.

**Tech Stack:** Python 3.11, NumPy, Streamlit, Plotly, pytest, Ruff

---

### Task 1: Define The Backend Publication Policy

**Files:**
- Modify: `config/config.py`
- Modify: `tests/config/test_config.py`

**Step 1: Write the failing test**

Add:

```python
def test_dlr_safety_factor_is_explicit_and_valid():
    assert config.DLR_SAFETY_FACTOR == 0.8
    assert 0.0 < config.DLR_SAFETY_FACTOR <= 1.0
```

**Step 2: Run test to verify it fails**

Run:

```bash
python3.11 -m pytest tests/config/test_config.py::test_dlr_safety_factor_is_explicit_and_valid -q
```

Expected: FAIL because `DLR_SAFETY_FACTOR` is not defined.

**Step 3: Write minimal implementation**

Add next to the other runtime constants:

```python
DLR_SAFETY_FACTOR = 0.8
```

Do not add a Streamlit control or environment override.

**Step 4: Run test to verify it passes**

Run:

```bash
python3.11 -m pytest tests/config/test_config.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add config/config.py tests/config/test_config.py
git commit -m "chore: define DLR publication safety factor"
```

### Task 2: Factor Steady And Transient Ratings Exactly Once

**Files:**
- Modify: `modules/dlr_pipeline.py`
- Modify: `tests/integration/test_dlr_pipeline.py`

**Step 1: Write failing steady publication test**

Using `_SpyThermalAdapter`, add a test that runs the pipeline and asserts:

```python
expected = np.full((2, 2), 800.0)
np.testing.assert_array_equal(result.max_currents, expected)
np.testing.assert_array_equal(result.thermal_result["max_currents"], expected)
np.testing.assert_array_equal(result.to_legacy_line_data()["max_currents"], expected)
assert result.thermal_result["safety_factor"] == 0.8
assert result.to_legacy_line_data()["safety_factor"] == 0.8
```

**Step 2: Write failing transient publication tests**

Create a test adapter that returns raw steady `1000 A` and raw transient `1200 A`. Assert:

```python
np.testing.assert_array_equal(result.max_currents, np.full((2, 2), 800.0))
np.testing.assert_array_equal(
    result.thermal_result["transient_result"]["max_currents"],
    np.full((2, 2), 960.0),
)
```

Strengthen the existing transient failure test to assert that its fallback is `800 A`, not `640 A`.

**Step 3: Run tests to verify they fail**

Run the new steady, transient-success, and fallback tests directly.

Expected: FAIL because the current pipeline publishes raw values.

**Step 4: Implement a single publication helper**

Import `DLR_SAFETY_FACTOR` and add a private helper:

```python
def _publish_dlr_currents(values: Any) -> np.ndarray:
    currents = np.asarray(values, dtype=float)
    if not np.isfinite(currents).all() or np.any(currents < 0.0):
        raise ValueError("DLR 额定值必须为有限非负值")
    return currents * DLR_SAFETY_FACTOR
```

In `DlrPipeline.run`:

1. Preserve raw `steady_result` for transient calculation.
2. Set `max_currents = _publish_dlr_currents(raw_steady_currents)`.
3. Replace `thermal_result["max_currents"]` with a factored copy and add `thermal_result["safety_factor"]`.
4. Factor a successful transient array before storing it.
5. On transient failure, store the already-factored steady array without another multiplication.
6. Add `safety_factor` to the legacy projection.

Do not alter `_dlr_input_hash`, `LongFrameThermalAdapter`, or `modules/thermal_engine.py`.

**Step 5: Run affected tests**

Run:

```bash
python3.11 -m pytest tests/integration/test_dlr_pipeline.py -q
python3.11 -m pytest tests/integration/test_weather_ai_dlr_e2e.py -q
python3.11 -m pytest tests/modules/test_thermal_engine_ieee738.py tests/modules/test_thermal_transient.py -q
```

Expected: PASS; the IEEE reference suite remains unchanged.

**Step 6: Commit**

```bash
git add modules/dlr_pipeline.py tests/integration/test_dlr_pipeline.py
git commit -m "feat: publish DLR ratings with safety factor"
```

### Task 3: Verify Sag Receives Only Published Ratings

**Files:**
- Modify: `tests/integration/test_sag_snapshot_bridge.py`

**Step 1: Write the integration test**

Build a real `DlrPipeline` result using a deterministic adapter, convert it with `to_legacy_line_data`, publish a sag snapshot, and assert every `snapshot.original_currents` value is `800 A` while the adapter supplied `1000 A`.

Also assert that mutating the legacy result after publication does not change the snapshot.

**Step 2: Run test and inspect result**

Run the test directly. It should pass only after Task 2; this is an integration proof rather than a new production change.

**Step 3: Run sag regression suite**

```bash
python3.11 -m pytest \
  tests/integration/test_sag_snapshot_bridge.py \
  tests/integration/test_sag_post_validation_e2e.py \
  tests/modules/test_sag_validation.py -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tests/integration/test_sag_snapshot_bridge.py
git commit -m "test: verify sag consumes factored DLR ratings"
```

### Task 4: Remove Static Baseline Runtime And Helper Paths

**Files:**
- Modify: `dispatch_app_st.py`
- Modify: `modules/visualization.py`
- Modify: `tests/modules/test_visualization.py`
- Modify: `tests/pages/test_streamlit_compatibility.py`

**Step 1: Write failing helper test**

Replace the old visualization test with:

```python
def test_build_line_rating_figure_contains_only_dynamic_rating():
    fig = build_line_rating_figure(
        timestamps=pd.to_datetime(["2025-12-10 00:00", "2025-12-10 01:00"]),
        dynamic_current=[640, 656],
    )
    assert len(fig.data) == 1
    assert fig.data[0].name == "动态额定值"
```

**Step 2: Write failing page-source contract**

Add a test that reads `dispatch_app_st.py` and rejects these runtime tokens:

```python
for token in (
    "static_val",
    "静态额定值（基准）",
    "对比静态",
    "静态额定值 (",
    "增容空间",
):
    assert token not in source
```

**Step 3: Run tests to verify they fail**

Run both direct tests. Expected: helper call fails because a static argument is required, and page source still contains static baseline tokens.

**Step 4: Remove the runtime baseline**

In `dispatch_app_st.py`:

- delete `static_p` and `static_val` calculation;
- change four summary columns to three;
- remove gain deltas from minimum and average metrics;
- remove the static line and filled “增容空间” trace;
- keep the dynamic line, three metrics, and all other page sections unchanged.

In `modules/visualization.py`, change the helper signature to:

```python
def build_line_rating_figure(timestamps, dynamic_current):
```

and emit one trace named `动态额定值`.

**Step 5: Run page and helper tests**

```bash
python3.11 -m pytest tests/modules/test_visualization.py tests/pages -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add dispatch_app_st.py modules/visualization.py tests/modules/test_visualization.py tests/pages/test_streamlit_compatibility.py
git commit -m "refactor: remove static rating baseline"
```

### Task 5: Documentation And Final Acceptance

**Files:**
- Modify: `README.md`
- Modify: `docs/TESTING.md`

**Step 1: Update documentation**

Document that:

- IEEE 738 raw reference calculations stay unscaled;
- all published steady/transient DLR values use `0.8`;
- failed transient fallback is factored once;
- no static baseline is displayed;
- the factor is backend-only and also reaches sag snapshots.

**Step 2: Run complete automated gates**

```bash
python3.11 -m pytest -q
python3.11 -m pytest tests/modules/test_thermal_engine_ieee738.py tests/modules/test_thermal_transient.py -q
python3.11 -m pytest tests/integration -q
python3.11 -m ruff check --target-version py311 .
python3.11 -m py_compile modules/*.py utils/*.py config/*.py dispatch_app_st.py thermal_functions.py pages/弧垂后验证.py
git diff --check
```

Expected: all commands exit `0`.

**Step 3: Browser acceptance**

Run Streamlit and verify at desktop `1440×900` and mobile `390×844`:

- the main page shows only minimum, maximum, and average dynamic ratings;
- no static baseline metric, line, comparison percentage, or fill remains;
- existing controls and plots do not overlap;
- a completed calculation publishes ratings equal to `0.8` times the deterministic raw backend probe;
- the sag page consumes the same published values;
- browser console has no warning/error and terminal has no new application exception.

**Step 4: Request independent review**

Review the branch against `fa0f259`, fix all Critical/Important findings with TDD, and rerun the full gates.

**Step 5: Commit documentation and acceptance tests**

```bash
git add README.md docs/TESTING.md
git commit -m "docs: document factored DLR publication"
```
