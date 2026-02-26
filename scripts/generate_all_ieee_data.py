"""
Generate all data needed for IEEE paper sections (v2 — ALERT-only fair comparison).

Changes from v1:
- Main MAS metric uses state=='ALERT' only (same as baseline) for fair comparison.
- Secondary "early-warning coverage" (ALERT+SUSPECTED) reported separately.
- Uses detection_delay = first_alarm_step − first_flood_step (positive = late).
- Adds extreme_dry as 4th hero scenario to show mid-sim flood onset.
- Saves baseline_results.json to FINAL_RESULTS/experiments/
- Saves Table_hero_summary.csv/.md to outputs/figures/
"""
import sys, json, yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.model import FloodModel, load_config
from baseline.threshold import ZonedThresholdBaseline
from joblib import load as jload

ROOT = Path(__file__).parent.parent
config = load_config(str(ROOT / 'configs' / 'default.yaml'))
with open(ROOT / 'configs' / 'scenarios.yaml') as f:
    sc_cfg = yaml.safe_load(f)

scenario_map = {s['name']: s for s in sc_cfg['scenarios']}
ml_model = jload(str(ROOT / 'FINAL_RESULTS' / 'model' / 'final_model.pkl'))

HEROES = ['normal_wet', 'extreme_wet', 'extreme_dropout_50', 'extreme_dry']
STEPS = 300
SIM_DIR = ROOT / 'FINAL_RESULTS' / 'simulations'

# ── helpers ──────────────────────────────────────────────────
def alarm_toggles(series):
    """Count 0→1 and 1→0 transitions in a boolean/int series."""
    s = series.astype(int).values
    return int(np.sum(np.abs(np.diff(s))))

def det_metrics(logs, alert_states):
    """Compute TP/FP/FN/TN/P/R/F1 for given alert states."""
    yt = logs['ground_truth_flooded'].astype(int).values
    yp = logs['state'].apply(lambda x: 1 if x in alert_states else 0).values
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else None
    r = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = 2 * p * r / (p + r) if (p and r and (p + r) > 0) else None
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'precision': round(p, 4) if p is not None else None,
            'recall': round(r, 4) if r is not None else None,
            'f1': round(f1, 4) if f1 is not None else None}

# ── main loop ────────────────────────────────────────────────
all_results = {}
table_rows = []

for name in HEROES:
    sc = scenario_map[name]
    sc_config = json.loads(json.dumps(config))  # deep copy
    if 'dropout_rate' in sc:
        sc_config['sensors']['dropout_rate'] = sc['dropout_rate']
    if 'noise_level' in sc:
        nl = sc_cfg.get('noise_levels', {})
        if sc['noise_level'] in nl:
            sc_config['sensors']['noise_std'] = nl[sc['noise_level']]['sensor_noise_std']

    soil_init = sc.get('soil_saturation_init', 0.3)
    rain_type = sc.get('rainfall_type', 'normal')

    # Run MAS
    mas = FloodModel(sc_config, ml_model=ml_model, seed=42)
    mas.reset(soil_init)
    mas.generate_random_rainfall(rain_type)

    # Run Baseline on same sim
    bl = ZonedThresholdBaseline(
        num_zones=sc_config['simulation']['num_zones'], config=sc_config)

    for step in range(STEPS):
        mas.step()
        zr = {}
        for zid, edge in mas.edges.items():
            zr[zid] = {
                'water': edge.current_features.get('water_mean_5', 0),
                'rain': edge.current_features.get('rain_sum_20', 0) / 20
            }
        bl.update(zr, step)

    mas_logs = mas.get_logs()
    coord_logs = mas.get_coordinator_logs()

    # Save hero logs for extreme_dry (new scenario)
    if name == 'extreme_dry':
        mas_logs.to_parquet(str(SIM_DIR / f'hero_{name}.parquet'), index=False)
        coord_logs.to_parquet(str(SIM_DIR / f'hero_{name}_coordinator.parquet'), index=False)
        print(f"  Saved hero logs for {name}")

    # ── baseline logs ──
    bl_records = []
    for zid in range(sc_config['simulation']['num_zones']):
        zh = bl.get_zone_history(zid)
        for _, row in zh.iterrows():
            flood_status = False
            zone_mas = mas_logs[(mas_logs['zone_id'] == zid) &
                                (mas_logs['step'] == row['step'])]
            if len(zone_mas) > 0:
                flood_status = bool(zone_mas.iloc[0]['ground_truth_flooded'])
            bl_records.append({
                'step': int(row['step']), 'zone_id': zid,
                'state': row['state'],
                'ground_truth_flooded': flood_status
            })
    bl_logs = pd.DataFrame(bl_records)

    # ── MAIN detection metrics: ALERT-only for both (fair comparison) ──
    mas_det_alert = det_metrics(mas_logs, ['ALERT'])
    bl_det_alert = det_metrics(bl_logs, ['ALERT'])

    # ── SECONDARY: ALERT+SUSPECTED coverage for MAS (early-warning) ──
    mas_det_early = det_metrics(mas_logs, ['ALERT', 'SUSPECTED'])

    # ── alarm toggles (fair stability) ──
    mas_toggles = alarm_toggles(coord_logs['global_alarm']) if len(coord_logs) > 0 else 0

    bl_global = bl_logs.groupby('step')['state'].apply(
        lambda s: int(any(x == 'ALERT' for x in s))).reset_index(name='alarm')
    bl_toggles = alarm_toggles(bl_global['alarm'])

    # ── per-zone ALERT toggles ──
    mas_zone_toggles = {}
    for zid in sorted(mas_logs['zone_id'].unique()):
        z = mas_logs[mas_logs['zone_id'] == zid].sort_values('step')
        is_alert = (z['state'] == 'ALERT').astype(int)
        mas_zone_toggles[int(zid)] = int(np.sum(np.abs(np.diff(is_alert.values))))

    bl_zone_toggles = {}
    for zid in sorted(bl_logs['zone_id'].unique()):
        z = bl_logs[bl_logs['zone_id'] == zid].sort_values('step')
        is_alert = (z['state'] == 'ALERT').astype(int)
        bl_zone_toggles[int(zid)] = int(np.sum(np.abs(np.diff(is_alert.values))))

    # ── first flood step ──
    first_flood_step = None
    for zid in sorted(mas_logs['zone_id'].unique()):
        z = mas_logs[mas_logs['zone_id'] == zid].sort_values('step')
        flooded = z[z['ground_truth_flooded'] == True]
        if len(flooded) > 0:
            fs = int(flooded['step'].min())
            if first_flood_step is None or fs < first_flood_step:
                first_flood_step = fs

    # First global alarm step (MAS)
    if len(coord_logs) > 0:
        alarmed = coord_logs[coord_logs['global_alarm'] == True]
        mas_first_alarm = int(alarmed['step'].min()) if len(alarmed) > 0 else None
    else:
        mas_first_alarm = None

    # Baseline first alarm step
    bl_alarm_rows = bl_global[bl_global['alarm'] == 1]
    bl_first_alarm = int(bl_alarm_rows['step'].min()) if len(bl_alarm_rows) > 0 else None

    # ── detection_delay = first_alarm_step − first_flood_step ──
    # positive = alarm came after flood (late detection)
    # negative = alarm came before flood (early warning)
    # None = no flood or no alarm
    mas_delay = None
    bl_delay = None
    if first_flood_step is not None:
        if mas_first_alarm is not None:
            mas_delay = mas_first_alarm - first_flood_step
        if bl_first_alarm is not None:
            bl_delay = bl_first_alarm - first_flood_step

    # ── mean risk ──
    mean_risk = round(float(mas_logs['risk'].mean()), 4)

    # ── count risk spikes above TH_UP ──
    th_up = sc_config['guardrails']['TH_UP']
    risk_spikes_above_thup = int((mas_logs['risk'] > th_up).sum())

    # ── assemble ──
    entry = {
        'scenario': name,
        'mas_detection_alert_only': mas_det_alert,
        'baseline_detection': bl_det_alert,
        'mas_early_warning_coverage': mas_det_early,
        'mas_global_alarm_toggles': mas_toggles,
        'baseline_global_alarm_toggles': bl_toggles,
        'mas_zone_alert_toggles': mas_zone_toggles,
        'baseline_zone_alert_toggles': bl_zone_toggles,
        'first_flood_step': first_flood_step,
        'mas_first_global_alarm_step': mas_first_alarm,
        'baseline_first_alarm_step': bl_first_alarm,
        'mas_detection_delay': mas_delay,
        'baseline_detection_delay': bl_delay,
        'mean_risk': mean_risk,
        'risk_spikes_above_TH_UP': risk_spikes_above_thup,
        'mas_alert_steps': int((mas_logs['state'] == 'ALERT').sum()),
        'mas_suspected_steps': int((mas_logs['state'] == 'SUSPECTED').sum()),
        'baseline_alert_steps': int((bl_logs['state'] == 'ALERT').sum()),
        'total_zone_steps': len(mas_logs),
        'total_flooded_zone_steps': int(mas_logs['ground_truth_flooded'].sum())
    }
    all_results[name] = entry
    print(f"  {name}: done")

    # Row for updated hero summary table (main = ALERT-only)
    table_rows.append({
        'scenario': name,
        'mean_risk': mean_risk,
        'mas_alert_steps': entry['mas_alert_steps'],
        'bl_alert_steps': entry['baseline_alert_steps'],
        'mas_first_alarm': mas_first_alarm,
        'bl_first_alarm': bl_first_alarm,
        'first_flood_step': first_flood_step,
        'mas_detection_delay': mas_delay,
        'bl_detection_delay': bl_delay,
        'mas_alarm_toggles': mas_toggles,
        'bl_alarm_toggles': bl_toggles,
        'mas_f1': mas_det_alert['f1'],
        'bl_f1': bl_det_alert['f1'],
        'mas_precision': mas_det_alert['precision'],
        'mas_recall': mas_det_alert['recall'],
        'bl_precision': bl_det_alert['precision'],
        'bl_recall': bl_det_alert['recall']
    })

# ── save baseline_results.json ──
out_dir = ROOT / 'FINAL_RESULTS' / 'experiments'
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'baseline_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {out_dir / 'baseline_results.json'}")

# ── save updated Table_hero_summary ──
tdf = pd.DataFrame(table_rows)
csv_path = ROOT / 'outputs' / 'figures' / 'Table_hero_summary.csv'
tdf.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

md_path = ROOT / 'outputs' / 'figures' / 'Table_hero_summary.md'
with open(md_path, 'w') as f:
    f.write(tdf.to_markdown(index=False))
    f.write('\n')
print(f"Saved: {md_path}")

# ── print summary ──
print("\n=== RESULTS AT A GLANCE (ALERT-only, fair comparison) ===")
for name, r in all_results.items():
    m = r['mas_detection_alert_only']
    b = r['baseline_detection']
    print(f"\n--- {name} ---")
    print(f"  MAS (ALERT-only): P={m['precision']}  R={m['recall']}  F1={m['f1']}")
    print(f"  Baseline         : P={b['precision']}  R={b['recall']}  F1={b['f1']}")
    print(f"  First flood step : {r['first_flood_step']}")
    print(f"  MAS first alarm  : {r['mas_first_global_alarm_step']}  BL: {r['baseline_first_alarm_step']}")
    print(f"  MAS det. delay   : {r['mas_detection_delay']}  BL: {r['baseline_detection_delay']}")
    print(f"  MAS alarm toggles: {r['mas_global_alarm_toggles']}  BL: {r['baseline_global_alarm_toggles']}")
    ew = r['mas_early_warning_coverage']
    print(f"  MAS early-warning (ALERT+SUSPECTED): P={ew['precision']}  R={ew['recall']}  F1={ew['f1']}")
