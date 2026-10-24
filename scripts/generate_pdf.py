"""Generate StartUP.pdf from docs/TECHNICAL_REPORT.md using headless Microsoft Edge."""

import os
import subprocess
import markdown

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
md_path = os.path.join(root_dir, "docs", "TECHNICAL_REPORT.md")
startup_md_path = os.path.join(root_dir, "docs", "StartUP.md")
html_path = os.path.join(root_dir, "docs", "StartUP.html")
pdf_path = os.path.join(root_dir, "docs", "StartUP.pdf")

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace header metadata with StartUP team branding
text_startup = text.replace(
    "**Pipeline Version:** `v2_optimized_pipeline`",
    "**Team Name:** StartUP\n**Pipeline Version:** `v2_optimized_pipeline`"
)

with open(startup_md_path, "w", encoding="utf-8") as f:
    f.write(text_startup)

html_content = markdown.markdown(text_startup, extensions=["tables", "fenced_code"])

styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>StartUP - More Than A Hack 2027 Technical Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #24292e;
        max-width: 850px;
        margin: 40px auto;
        padding: 0 20px;
    }}
    h1 {{
        font-size: 20pt;
        border-bottom: 2px solid #0366d6;
        padding-bottom: 8px;
        color: #0b2240;
    }}
    h2 {{
        font-size: 14pt;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 6px;
        margin-top: 24px;
        color: #1a365d;
    }}
    h3 {{
        font-size: 12pt;
        margin-top: 18px;
        color: #2b6cb0;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 16px 0;
        font-size: 9.5pt;
    }}
    th, td {{
        border: 1px solid #d1d5da;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{
        background-color: #f6f8fa;
        font-weight: 600;
    }}
    tr:nth-child(even) {{
        background-color: #fbfcfd;
    }}
    pre, code {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 9pt;
        background-color: #f6f8fa;
        border-radius: 3px;
    }}
    code {{
        padding: 2px 4px;
    }}
    pre {{
        padding: 12px;
        overflow-x: auto;
        border: 1px solid #e1e4e8;
    }}
    hr {{
        height: 1px;
        background-color: #e1e4e8;
        border: none;
        margin: 24px 0;
    }}
    @page {{
        margin: 20mm;
    }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(styled_html)

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if os.path.exists(edge_exe):
    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    print(f"Converting HTML to PDF via Microsoft Edge Headless: {pdf_path}")
    subprocess.run(cmd, check=True)
    print(f"[OK] Generated: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
else:
    print(f"[WARN] Edge not found at {edge_exe}")
