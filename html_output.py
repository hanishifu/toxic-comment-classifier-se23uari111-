# src/html_output.py
import pandas as pd
import os
import webbrowser
import html

INPUT_CSV = "data/toxic_results.csv"
OUTPUT_HTML = "outputs/toxic_results.html"

def generate_html(input_csv=INPUT_CSV, output_html=OUTPUT_HTML, auto_open=True):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"{input_csv} not found. Run classify.py first.")

    df = pd.read_csv(input_csv)

    # Ensure columns exist
    for c in ["text", "toxic_score", "neutral_score", "prediction"]:
        if c not in df.columns:
            df[c] = ""

    # Escape HTML in text to avoid broken HTML
    df["text_html"] = df["text"].astype(str).apply(lambda t: html.escape(t).replace("\n", "<br>"))

    # Build rows with classes for styling
    rows_html = []
    for _, r in df.iterrows():
        pred = str(r["prediction"])
        toxic = float(r.get("toxic_score", 0) or 0)
        neutral = float(r.get("neutral_score", 0) or 0)
        # decide row class
        row_class = "toxic" if pred.lower().startswith("toxic") else "safe"
        # formatted scores
        toxic_s = f"{toxic:.3f}"
        neutral_s = f"{neutral:.3f}"
        text_cell = r["text_html"]
        rows_html.append(f"""
        <tr class="{row_class}">
          <td style="vertical-align:top; width:6%;">{int(_)+1}</td>
          <td style="vertical-align:top; width:44%;"><div class="text-cell">{text_cell}</div></td>
          <td style="text-align:center; width:15%;">{toxic_s}</td>
          <td style="text-align:center; width:15%;">{neutral_s}</td>
          <td style="text-align:center; width:10%;"><strong>{pred}</strong></td>
        </tr>
        """)

    rows_html_str = "\n".join(rows_html)

    # HTML template (clean design)
    html_content = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Toxic Classification Results</title>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <style>
        :root {{
          --bg:#0b1020;
          --card:#0f1724;
          --muted:#9aa4b2;
          --accent:#06b6d4;
          --danger:#ef4444;
          --good:#10b981;
        }}
        body{{font-family:Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial; background:#071022; color:#e6eef6; margin:0; padding:28px;}}
        .container{{max-width:1100px; margin:0 auto;}}
        header{{display:flex; align-items:center; gap:14px; margin-bottom:18px;}}
        header h1{{margin:0; font-size:20px; letter-spacing:0.2px;}}
        .card{{background:linear-gradient(180deg,#071022, #0b1424); border:1px solid rgba(255,255,255,0.03); padding:14px; border-radius:12px; box-shadow:0 6px 30px rgba(2,6,23,0.6);}}
        .summary{{display:flex; gap:18px; align-items:center; margin-bottom:12px;}}
        .pill{{background:var(--card); padding:6px 10px; border-radius:999px; color:var(--muted); font-size:13px;}}
        table{{width:100%; border-collapse:collapse; margin-top:12px;}}
        th, td{{padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.04); font-size:14px;}}
        th{{text-align:left; background:transparent; color:var(--muted); font-weight:600; font-size:13px;}}
        tr.toxic td{{background: linear-gradient(90deg, rgba(239,68,68,0.06), rgba(239,68,68,0.02));}}
        tr.safe td{{background: transparent;}}
        .text-cell{{white-space:pre-wrap; color:#e8f0ff;}}
        .pred-tox{{color:var(--danger); font-weight:700;}}
        .pred-safe{{color:var(--good); font-weight:700;}}
        footer{{margin-top:14px; color:var(--muted); font-size:13px;}}
        @media (max-width:900px){{
          th, td{{font-size:12px; padding:8px;}}
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <header>
          <div class="card" style="padding:10px 14px;">
            <h1>Toxic Comment Classification — Results</h1>
            <div style="margin-top:6px; color:var(--muted); font-size:13px;">Model: <strong>s-nlp/roberta_toxicity_classifier</strong></div>
          </div>
        </header>

        <div class="card">
          <div class="summary">
            <div class="pill">Rows: <strong>{len(df)}</strong></div>
            <div class="pill">Toxic count: <strong>{(df['prediction']=='Toxic').sum() if 'prediction' in df.columns else 0}</strong></div>
            <div class="pill">Safe count: <strong>{(df['prediction']!='Toxic').sum() if 'prediction' in df.columns else 0}</strong></div>
          </div>

          <table>
            <thead>
              <tr>
                <th style="width:6%;">#</th>
                <th>Text</th>
                <th style="width:15%; text-align:center;">Toxic score</th>
                <th style="width:15%; text-align:center;">Neutral score</th>
                <th style="width:10%; text-align:center;">Prediction</th>
              </tr>
            </thead>
            <tbody>
              {rows_html_str}
            </tbody>
          </table>

          <footer>
            Tip: Red rows are flagged as <strong>toxic</strong>. Use this for moderation or filtering.
          </footer>
        </div>
      </div>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Saved HTML to:", output_html)
    if auto_open:
        webbrowser.open("file://" + os.path.abspath(output_html))

if __name__ == "__main__":
    generate_html()
