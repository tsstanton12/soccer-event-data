#!/usr/bin/env python3

import argparse
import base64
import csv
import html
import json
import mimetypes
from pathlib import Path


def image_data_url(path):
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_html", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output_html).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    items = []
    for index, row in enumerate(rows):
        image_path = Path(row["debug_image"])
        if not image_path.is_absolute():
            candidates = [Path.cwd() / image_path, manifest_path.parent / image_path]
            image_path = next((path for path in candidates if path.exists()), candidates[0])
        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue
        items.append({
            "index": index,
            "image": image_data_url(image_path),
            "row": row,
        })

    fieldnames = list(rows[0].keys()) if rows else []
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field Segmentation Review</title>
<style>
body{{margin:0;background:#121212;color:#eee;font:16px system-ui,sans-serif}}header{{position:sticky;top:0;background:#1e1e1eee;padding:12px 18px;display:flex;gap:12px;align-items:center;z-index:2}}button{{padding:9px 14px;border:1px solid #666;border-radius:6px;background:#292929;color:#fff;cursor:pointer}}button.active{{background:#087f23;border-color:#45d36b}}#stage{{padding:16px;text-align:center}}img{{max-width:100%;max-height:78vh}}#details{{margin:10px auto;max-width:1200px;text-align:left;color:#ccc}}.spacer{{flex:1}}kbd{{background:#444;padding:2px 6px;border-radius:4px}}</style></head>
<body><header><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><span id="count"></span><span class="spacer"></span>
<button data-label="good" onclick="label('good')"><kbd>G</kbd> Good</button><button data-label="too_wide" onclick="label('too_wide')"><kbd>W</kbd> Too wide</button><button data-label="too_tight" onclick="label('too_tight')"><kbd>T</kbd> Too tight</button><button data-label="missed" onclick="label('missed')"><kbd>M</kbd> Missed</button><button onclick="exportCsv()">Export CSV</button></header>
<main id="stage"><img id="image"><div id="details"></div></main>
<script>
const items={json.dumps(items)}; const fields={json.dumps(fieldnames)}; let current=0; const key='field-review:{html.escape(str(manifest_path))}'; const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
items.forEach(x=>{{if(saved[x.index]!==undefined)x.row.review=saved[x.index]}});
function show(){{if(!items.length)return;const x=items[current];document.getElementById('image').src=x.image;document.getElementById('count').textContent=`${{current+1}} / ${{items.length}}`;document.getElementById('details').textContent=`${{x.row.video}} | frame ${{x.row.frame}} | confidence ${{x.row.confidence}} | review: ${{x.row.review||'unreviewed'}}`;document.querySelectorAll('[data-label]').forEach(b=>b.classList.toggle('active',b.dataset.label===x.row.review));}}
function move(n){{current=Math.max(0,Math.min(items.length-1,current+n));show()}}function label(v){{items[current].row.review=v;saved[items[current].index]=v;localStorage.setItem(key,JSON.stringify(saved));show();move(1)}}
function esc(v){{v=String(v??'');return /[",\n]/.test(v)?'"'+v.replaceAll('"','""')+'"':v}}function exportCsv(){{const csv=[fields.join(','),...items.map(x=>fields.map(f=>esc(x.row[f])).join(','))].join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='evaluation_manifest_reviewed.csv';a.click()}}
document.addEventListener('keydown',e=>{{if(e.key==='ArrowRight'||e.key===' ')move(1);if(e.key==='ArrowLeft')move(-1);if(e.key.toLowerCase()==='g')label('good');if(e.key.toLowerCase()==='w')label('too_wide');if(e.key.toLowerCase()==='t')label('too_tight');if(e.key.toLowerCase()==='m')label('missed')}});show();
</script></body></html>"""
    output_path.write_text(page)
    print(f"Embedded {len(items)} images")
    print(f"Review gallery: {output_path}")


if __name__ == "__main__":
    main()
