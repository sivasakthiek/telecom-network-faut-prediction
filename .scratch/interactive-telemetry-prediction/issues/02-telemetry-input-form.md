# 02 — Interactive Telemetry Configuration Form & Dynamic Volume Steppers

**What to build:** A dedicated top-level "Live Telemetry Simulator" tab containing structured form inputs for all required network incident parameters, including searchable dropdowns, custom location text input fallback, and dynamic volume steppers for active log features.

**Blocked by:** 01 — Vectorized Feature Alignment & Fast Inference Pipeline

**Status:** done

- [x] Add top-level tab switcher to the NOC command center for "⚡ Live Telemetry Simulator".
- [x] Render searchable location dropdown with 929 naturally sorted locations and a toggle for entering custom unlisted locations.
- [x] Provide categorical selectbox for reported severity type (1 to 5) and multiselects for resource types and network event types.
- [x] Provide searchable multiselect for 386 log features with dynamically generated numeric volume steppers for each selected signal.
