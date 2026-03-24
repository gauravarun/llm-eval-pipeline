# reporting/dashboard.py
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from models.interface import load_all_checkpoint_results
from reporting.aggregator import build_comparison_table, build_failure_matrix

DASHBOARD_PATH = Path("checkpoints/dashboard.html")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Evaluation Pipeline — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0f1117; color: #e2e8f0; line-height: 1.6; }}
  .header {{ background: #1a1d2e; border-bottom: 1px solid #2d3748;
             padding: 1.5rem 2rem; }}
  .header h1 {{ font-size: 1.25rem; font-weight: 600; color: #fff; }}
  .header p  {{ font-size: 0.85rem; color: #718096; margin-top: 0.2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 1rem; padding: 1.5rem 2rem; }}
  .card {{ background: #1a1d2e; border: 1px solid #2d3748;
           border-radius: 10px; padding: 1.25rem; }}
  .card .label {{ font-size: 0.75rem; color: #718096;
                  text-transform: uppercase; letter-spacing: 0.05em; }}
  .card .value {{ font-size: 1.75rem; font-weight: 600;
                  color: #fff; margin-top: 0.25rem; }}
  .card .sub   {{ font-size: 0.75rem; color: #718096; margin-top: 0.2rem; }}
  .section {{ padding: 0 2rem 2rem; }}
  .section h2 {{ font-size: 0.9rem; font-weight: 600; color: #a0aec0;
                 text-transform: uppercase; letter-spacing: 0.08em;
                 margin-bottom: 1rem; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  .chart-card {{ background: #1a1d2e; border: 1px solid #2d3748;
                 border-radius: 10px; padding: 1.25rem; }}
  .chart-card h3 {{ font-size: 0.8rem; color: #718096; margin-bottom: 1rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; padding: 0.5rem 0.75rem; color: #718096;
        font-weight: 500; border-bottom: 1px solid #2d3748; }}
  td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid #1e2130; }}
  tr:hover td {{ background: #1e2440; }}
  .pass  {{ color: #68d391; font-weight: 500; }}
  .fail  {{ color: #fc8181; font-weight: 500; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem;
            border-radius: 4px; font-size: 0.75rem; font-weight: 500; }}
  .badge-easy   {{ background: #1a3a2a; color: #68d391; }}
  .badge-medium {{ background: #3a2a1a; color: #f6ad55; }}
  .badge-hard   {{ background: #3a1a1a; color: #fc8181; }}
  .ci-bar {{ display: inline-block; width: 80px; height: 6px;
             background: #2d3748; border-radius: 3px;
             position: relative; vertical-align: middle; margin-left: 6px; }}
  .ci-fill {{ position: absolute; height: 100%; background: #4299e1;
              border-radius: 3px; }}
  @media (max-width: 700px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>LLM Evaluation Pipeline</h1>
  <p>Checkpoint comparison · Statistical analysis · Failure mode taxonomy</p>
</div>

<!-- Metric cards -->
<div class="grid">
  <div class="card">
    <div class="label">Checkpoints evaluated</div>
    <div class="value">{n_checkpoints}</div>
    <div class="sub">Versioned runs stored</div>
  </div>
  <div class="card">
    <div class="label">Total tasks</div>
    <div class="value">{n_tasks}</div>
    <div class="sub">Across {n_categories} categories</div>
  </div>
  <div class="card">
    <div class="label">Best pass rate</div>
    <div class="value">{best_pass}%</div>
    <div class="sub">{best_label}</div>
  </div>
  <div class="card">
    <div class="label">Lowest category</div>
    <div class="value">{worst_cat}</div>
    <div class="sub">avg score {worst_score}</div>
  </div>
</div>

<!-- Charts -->
<div class="section">
  <h2>Category scores by checkpoint</h2>
  <div class="charts">
    <div class="chart-card">
      <h3>Pass rate by category (all checkpoints)</h3>
      <canvas id="barChart" height="220"></canvas>
    </div>
    <div class="chart-card">
      <h3>Score distribution by difficulty</h3>
      <canvas id="diffChart" height="220"></canvas>
    </div>
  </div>
</div>

<!-- Checkpoint comparison table -->
<div class="section">
  <h2>Checkpoint comparison</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Checkpoint</th><th>Mean score</th><th>Pass rate</th>
          <th>95% CI</th><th>Std dev</th><th>Run at</th>
        </tr>
      </thead>
      <tbody>
        {checkpoint_rows}
      </tbody>
    </table>
  </div>
</div>

<!-- Category breakdown table -->
<div class="section">
  <h2>Category breakdown — baseline checkpoint</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Category</th><th>Avg score</th><th>Pass rate</th>
          <th>95% CI</th><th>CV (consistency)</th><th>Failed tasks</th>
        </tr>
      </thead>
      <tbody>
        {category_rows}
      </tbody>
    </table>
  </div>
</div>

<!-- Failure matrix -->
<div class="section">
  <h2>Task failure matrix</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Task ID</th><th>Category</th><th>Difficulty</th>
          {failure_headers}
        </tr>
      </thead>
      <tbody>
        {failure_rows}
      </tbody>
    </table>
  </div>
</div>

<script>
const chartData = {chart_data_json};

// Bar chart — pass rate by category
new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: chartData.categories,
    datasets: chartData.checkpoints.map((label, i) => ({{
      label,
      data: chartData.pass_rates[i],
      backgroundColor: ['#4299e1','#68d391','#f6ad55'][i % 3],
      borderRadius: 4,
      borderSkipped: false,
    }}))
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#a0aec0', font: {{ size: 11 }} }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#718096' }}, grid: {{ color: '#2d3748' }} }},
      y: {{ min: 0, max: 1, ticks: {{ color: '#718096',
            callback: v => (v*100).toFixed(0)+'%' }},
           grid: {{ color: '#2d3748' }} }}
    }}
  }}
}});

// Difficulty chart
new Chart(document.getElementById('diffChart'), {{
  type: 'bar',
  data: {{
    labels: ['easy', 'medium', 'hard'],
    datasets: chartData.checkpoints.map((label, i) => ({{
      label,
      data: chartData.diff_pass_rates[i],
      backgroundColor: ['#4299e1','#68d391','#f6ad55'][i % 3],
      borderRadius: 4,
      borderSkipped: false,
    }}))
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#a0aec0', font: {{ size: 11 }} }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#718096' }}, grid: {{ color: '#2d3748' }} }},
      y: {{ min: 0, max: 1, ticks: {{ color: '#718096',
            callback: v => (v*100).toFixed(0)+'%' }},
           grid: {{ color: '#2d3748' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""


def build_dashboard():
    all_results = load_all_checkpoint_results()
    if not all_results:
        print("No results found. Run python run.py first.")
        return

    summaries = [cr["summary"] for cr in all_results]
    comparison_rows = build_comparison_table(all_results)
    matrix = build_failure_matrix(all_results)

    # Metric card values
    n_checkpoints = len(summaries)
    n_tasks = summaries[0]["total_tasks"]
    n_categories = len(summaries[0]["category_breakdown"])
    best = max(summaries, key=lambda s: s["pass_rate"])
    best_pass = round(best["pass_rate"] * 100, 1)
    best_label = best["label"]

    cat_avgs = {
        cat: stats["avg_score"]
        for cat, stats in summaries[0]["category_breakdown"].items()
    }
    worst_cat = min(cat_avgs, key=cat_avgs.get)
    worst_score = round(cat_avgs[worst_cat], 3)

    # Checkpoint rows
    ckpt_rows_html = ""
    for s in summaries:
        ci = s["ci_95"]
        ci_width = ci[1] - ci[0]
        ci_start = ci[0]
        ckpt_rows_html += f"""
        <tr>
          <td>{s['label']}</td>
          <td>{s['mean_score']}</td>
          <td class="{'pass' if s['pass_rate'] >= 0.7 else 'fail'}">
            {round(s['pass_rate']*100,1)}%</td>
          <td>[{ci[0]}, {ci[1]}]
            <span class="ci-bar">
              <span class="ci-fill"
                style="left:{ci_start*100:.1f}%;width:{ci_width*100:.1f}%"></span>
            </span>
          </td>
          <td>{s['std']}</td>
          <td style="color:#718096;font-size:0.8rem">{s.get('run_at','')}</td>
        </tr>"""

    # Category rows (baseline = first checkpoint)
    cat_rows_html = ""
    for cat, stats in summaries[0]["category_breakdown"].items():
        ci = stats["ci_95"]
        cv_color = "pass" if stats["cv"] < 0.3 else ("fail" if stats["cv"] > 0.7 else "")
        failed = ", ".join(stats["failed_tasks"]) if stats["failed_tasks"] else "—"
        cat_rows_html += f"""
        <tr>
          <td>{cat}</td>
          <td>{stats['avg_score']}</td>
          <td class="{'pass' if stats['pass_rate'] >= 0.7 else 'fail'}">
            {round(stats['pass_rate']*100,0):.0f}%</td>
          <td>[{ci[0]}, {ci[1]}]</td>
          <td class="{cv_color}">{stats['cv']}</td>
          <td style="color:#718096;font-size:0.8rem">{failed}</td>
        </tr>"""

    # Failure matrix
    labels = [cr["summary"]["label"] for cr in all_results]
    failure_headers_html = "".join(f"<th>{l}</th>" for l in labels)
    failure_rows_html = ""
    for tid, data in sorted(matrix.items()):
        diff = data["difficulty"]
        cells = ""
        for label in labels:
            ckpt_data = data["checkpoints"].get(label, {})
            if ckpt_data:
                cells += (f'<td class="pass">PASS</td>'
                          if ckpt_data["pass"]
                          else f'<td class="fail">FAIL</td>')
            else:
                cells += "<td>—</td>"
        failure_rows_html += f"""
        <tr>
          <td style="font-family:monospace;font-size:0.8rem">{tid}</td>
          <td>{data['category']}</td>
          <td><span class="badge badge-{diff}">{diff}</span></td>
          {cells}
        </tr>"""

    # Chart data
    categories = list(summaries[0]["category_breakdown"].keys())
    diff_levels = ["easy", "medium", "hard"]
    chart_data = {
        "categories": categories,
        "checkpoints": [s["label"] for s in summaries],
        "pass_rates": [
            [summaries[i]["category_breakdown"][cat]["pass_rate"]
             for cat in categories]
            for i in range(len(summaries))
        ],
        "diff_pass_rates": [
            [summaries[i]["difficulty_breakdown"].get(d, {}).get("pass_rate", 0)
             for d in diff_levels]
            for i in range(len(summaries))
        ]
    }

    html = HTML_TEMPLATE.format(
        n_checkpoints=n_checkpoints,
        n_tasks=n_tasks,
        n_categories=n_categories,
        best_pass=best_pass,
        best_label=best_label,
        worst_cat=worst_cat,
        worst_score=worst_score,
        checkpoint_rows=ckpt_rows_html,
        category_rows=cat_rows_html,
        failure_headers=failure_headers_html,
        failure_rows=failure_rows_html,
        chart_data_json=json.dumps(chart_data)
    )

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"\n  Dashboard written to: {DASHBOARD_PATH.resolve()}")
    print(f"  Open in browser: file:///{DASHBOARD_PATH.resolve()}\n")
    return DASHBOARD_PATH


if __name__ == "__main__":
    build_dashboard()