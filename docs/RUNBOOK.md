# Runbook

Stub. Sections are filled in as the milestones that need them land.

## Configuration

- Copy `config/farmhub.example.toml` to `config/farmhub.toml` (gitignored).
- Validate: `uv run farmhub --config config/farmhub.toml config check`. It exits 2 on invalid
  config and prints the effective safety settings first.
- `dry_run` is ON unless something turns it off. When it is off, startup logs a warning.

## Home Assistant side (SPEC §3.4), filled in at M5

FarmHub cannot verify these from Python. Check each by hand before enabling any T2 tool:

- [ ] Every irrigation, watering and ventilation script enforces a maximum runtime inside
      the script itself (not in FarmHub). Record the value next to the tool's `max_runtime_s`.
- [ ] Heating and pump control exist only as deterministic HA automations and fixed intents.
- [ ] The HA user behind FarmHub's token: confirm whether "non-admin" actually restricts
      service calls. Assume it does not (docs/DECISIONS.md).
- [ ] Each valve and pump circuit has an independent hardware cutoff that does not route
      through Home Assistant (SPEC §10).
