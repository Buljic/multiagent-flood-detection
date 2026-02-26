"""Compute baseline vs MAS comparison for hero scenarios."""
import yaml, numpy as np, pandas as pd, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.model import FloodModel, load_config
from baseline.threshold import ZonedThresholdBaseline
from joblib import load as jload

config = load_config('configs/default.yaml')
with open('configs/scenarios.yaml') as f:
    sc_cfg = yaml.safe_load(f)

scenario_map = {s['name']: s for s in sc_cfg['scenarios']}
ml_model = jload('FINAL_RESULTS/model/final_model.pkl')

results = {}

for name in ['normal_wet', 'extreme_wet', 'extreme_dropout_50']:
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

    mas = FloodModel(sc_config, ml_model=ml_model, seed=42)
    mas.reset(soil_init)
    mas.generate_random_rainfall(rain_type)

    bl = ZonedThresholdBaseline(num_zones=sc_config['simulation']['num_zones'], config=sc_config)

    for step in range(300):
        mas.step()
        zr = {}
        for zid, edge in mas.edges.items():
            zr[zid] = {
                'water': edge.current_features.get('water_mean_5', 0),
                'rain': edge.current_features.get('rain_sum_20', 0) / 20
            }
        bl.update(zr, step)

    mas_logs = mas.get_logs()

    bl_records = []
    for zid in range(sc_config['simulation']['num_zones']):
        zh = bl.get_zone_history(zid)
        for _, row in zh.iterrows():
            bl_records.append({
                'step': row['step'], 'zone_id': zid,
                'state': row['state'],
                'ground_truth_flooded': mas.environment.is_flooded(zid)
            })
    bl_logs = pd.DataFrame(bl_records)

    def metrics(y_true, y_pred):
        tp = int(((y_pred==1)&(y_true==1)).sum())
        fp = int(((y_pred==1)&(y_true==0)).sum())
        fn = int(((y_pred==0)&(y_true==1)).sum())
        tn = int(((y_pred==0)&(y_true==0)).sum())
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        return {'P': round(p,4), 'R': round(r,4), 'F1': round(f1,4), 'TP':tp, 'FP':fp, 'FN':fn, 'TN':tn}

    mas_yt = mas_logs['ground_truth_flooded'].astype(int).values
    mas_yp = mas_logs['state'].apply(lambda x: 1 if x in ['ALERT','SUSPECTED'] else 0).values
    bl_yt = bl_logs['ground_truth_flooded'].astype(int).values
    bl_yp = bl_logs['state'].apply(lambda x: 1 if x=='ALERT' else 0).values

    mas_m = metrics(mas_yt, mas_yp)
    bl_m = metrics(bl_yt, bl_yp)

    mas_sc = sum(e.state_machine.state_changes for e in mas.edges.values())
    bl_sc = bl.get_total_state_changes()

    mas_fa = mas_logs[mas_logs['state']=='ALERT']['step']
    bl_fa = bl_logs[bl_logs['state']=='ALERT']['step']
    flood_rows = mas_logs[mas_logs['ground_truth_flooded']==True]
    first_flood = int(flood_rows['step'].min()) if len(flood_rows)>0 else None

    results[name] = {
        'mas': mas_m, 'baseline': bl_m,
        'mas_state_changes': mas_sc, 'bl_state_changes': bl_sc,
        'mas_first_alert': int(mas_fa.min()) if len(mas_fa)>0 else None,
        'bl_first_alert': int(bl_fa.min()) if len(bl_fa)>0 else None,
        'first_flood_step': first_flood,
        'total_flooded_steps': int(mas_yt.sum()),
        'total_steps': len(mas_yt)
    }
    print(f"{name}: done")

print("\n" + json.dumps(results, indent=2, default=str))
