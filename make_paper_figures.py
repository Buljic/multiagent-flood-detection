import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

HERO = [
    ("normal_wet", "hero_normal_wet.parquet", "hero_normal_wet_coordinator.parquet"),
    ("extreme_wet", "hero_extreme_wet.parquet", "hero_extreme_wet_coordinator.parquet"),
    ("extreme_dropout_50", "hero_extreme_dropout_50.parquet", "hero_extreme_dropout_50_coordinator.parquet"),
]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_required(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)

def plot_zone_timeline(df: pd.DataFrame, scenario_name: str, out_dir: str, th_up=0.6, th_down=0.4, zone_id=0):
    # očekivane kolone: zone_id, step, risk, state, ground_truth_flooded
    z = df[df["zone_id"] == zone_id].copy().sort_values("step")
    z["alarm"] = (z["state"] == "ALERT").astype(int)
    gt_col = "ground_truth_flooded" if "ground_truth_flooded" in z.columns else None

    fig, ax = plt.subplots()
    ax.plot(z["step"], z["risk"], linewidth=2, label="risk")
    ax.axhline(th_up, linestyle="--", label="TH_UP")
    ax.axhline(th_down, linestyle="--", label="TH_DOWN")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Risk probability")
    ax.set_title(f"Timeline - {scenario_name} - Zone {zone_id}")
    ax.legend()
    ax.grid(alpha=0.3)

    # drugi y za alarm + ground truth (0/1)
    ax2 = ax.twinx()
    ax2.plot(z["step"], z["alarm"], label="ALERT(state)", linewidth=1)
    if gt_col:
        ax2.plot(z["step"], z[gt_col].astype(int), label="ground_truth_flooded", linewidth=1)
    ax2.set_ylabel("Alarm / Ground truth")
    ax2.set_yticks([0, 1])

    fig.tight_layout()
    out_png = os.path.join(out_dir, f"Fig_timeline_{scenario_name}_zone{zone_id}.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # TXT objašnjenje
    out_txt = os.path.join(out_dir, f"Fig_timeline_{scenario_name}_zone{zone_id}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(
            f"- X osa: simulation step (vrijeme)\n"
            f"- Linija 'risk': ML vjerovatnoća poplave\n"
            f"- TH_UP/TH_DOWN: guardrails pragovi (hysteresis)\n"
            f"- ALERT(state): kada sistem uđe u ALERT\n"
            f"- ground_truth_flooded: stvarni flood event (ako postoji kolona)\n"
        )

def plot_global_timeline(coord: pd.DataFrame, scenario_name: str, out_dir: str):
    # očekivane kolone: step, global_risk, global_alarm, zones_in_alert:contentReference[oaicite:6]{index=6}
    c = coord.copy().sort_values("step")
    c["global_alarm_i"] = c["global_alarm"].astype(int)

    fig, ax = plt.subplots()
    ax.plot(c["step"], c["global_risk"], linewidth=2, label="global_risk")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Global risk")
    ax.set_title(f"Global Timeline - {scenario_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(c["step"], c["global_alarm_i"], linewidth=1, label="global_alarm")
    ax2.set_ylabel("Global alarm")
    ax2.set_yticks([0, 1])

    fig.tight_layout()
    out_png = os.path.join(out_dir, f"Fig_global_{scenario_name}.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    out_txt = os.path.join(out_dir, f"Fig_global_{scenario_name}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(
            f"- global_risk: agregirani rizik koordinatora\n"
            f"- global_alarm: globalna uzbuna (0/1)\n"
            f"- X osa: simulation step (vrijeme)\n"
        )

def save_summary_row(rows, scenario_name, df, coord):
    # per-scenario: mean risk, alert count, first alert step, first flood step, lead time
    mean_risk = float(df["risk"].mean())
    alert_steps = int((df["state"] == "ALERT").sum())
    # global alarm
    first_global_alarm = None
    if "global_alarm" in coord.columns:
        a = coord[coord["global_alarm"] == True]
        if len(a) > 0:
            first_global_alarm = int(a.iloc[0]["step"])
    # first flood (any zone)
    first_flood = None
    if "ground_truth_flooded" in df.columns:
        gt = df[df["ground_truth_flooded"] == True]
        if len(gt) > 0:
            first_flood = int(gt.sort_values("step").iloc[0]["step"])

    lead_time = None
    if first_global_alarm is not None and first_flood is not None:
        lead_time = first_flood - first_global_alarm  # >0 znači warning prije flood-a

    rows.append({
        "scenario": scenario_name,
        "mean_risk": mean_risk,
        "alert_steps_total": alert_steps,
        "first_global_alarm_step": first_global_alarm,
        "first_flood_step": first_flood,
        "lead_time_steps": lead_time
    })

def plot_confusion_from_report(report_path: str, out_dir: str):
    if not os.path.exists(report_path):
        return
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    cm = rep.get("confusion_matrix", None)
    if not cm:
        return

    fig, ax = plt.subplots()
    ax.imshow(cm)  # default colormap
    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    out_png = os.path.join(out_dir, "Fig_confusion_testset.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    out_txt = os.path.join(out_dir, "Fig_confusion_testset.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("- Confusion matrix iz final_report.json (test set)\n- [[TN, FP],[FN, TP]]\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir", default="FINAL_RESULTS/simulations", help="Folder sa hero_*.parquet")
    ap.add_argument("--model_report", default="FINAL_RESULTS/model/final_report.json", help="final_report.json")
    ap.add_argument("--out_dir", default="outputs/figures", help="gdje snimiti PNG/TXT")
    ap.add_argument("--zone", type=int, default=0, help="koju zonu crtati za zone timeline")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # 1) figure iz test reporta
    plot_confusion_from_report(args.model_report, args.out_dir)

    # 2) hero scenariji: zone timeline + global timeline + summary
    summary_rows = []
    for scenario_name, zone_file, coord_file in HERO:
        df = load_required(os.path.join(args.sim_dir, zone_file))
        coord = load_required(os.path.join(args.sim_dir, coord_file))

        plot_zone_timeline(df, scenario_name, args.out_dir, zone_id=args.zone)
        plot_global_timeline(coord, scenario_name, args.out_dir)
        save_summary_row(summary_rows, scenario_name, df, coord)

    summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, "Table_hero_summary.csv")
    summary.to_csv(summary_csv, index=False)

    # i kratki markdown za paper
    md_path = os.path.join(args.out_dir, "Table_hero_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary.to_markdown(index=False))

    print(f"Saved figures and tables to: {args.out_dir}")

if __name__ == "__main__":
    main()
