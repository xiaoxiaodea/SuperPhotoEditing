"""Build a searchable private gallery and a complete-only runtime lesson index."""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageOps

from superphoto.master_studies import StudyAnalysis, workflow_lessons
from superphoto.ingest import image_to_srgb
from study_hasselblad import write_json


def collect(folder):
    inventory = json.loads((folder / "inventory.json").read_text(encoding="utf-8"))
    review_path = folder / "visual-review.json"
    reviewed = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}
    records, pending, invalid = [], [], []
    for source in inventory:
        path = folder / "records" / (source["id"] + ".json")
        if not path.is_file():
            pending.append(source)
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("schema_version") != 3 or record.get("status") != "complete" or record["sha256"] != source["sha256"]:
                pending.append(source)
                continue
            StudyAnalysis.model_validate(record["analysis"])
            if source["id"] in reviewed:
                correction = reviewed[source["id"]]
                if correction["sha256"] != source["sha256"]:
                    raise ValueError("Visual review belongs to different image bytes")
                record["analysis"] = StudyAnalysis.model_validate({**record["analysis"], **correction["patch"]}).model_dump()
                record["visual_review"] = {k: correction[k] for k in ("reviewed_at", "note")}
        except (ValueError, KeyError) as exc:
            invalid.append({"id": source["id"], "error": str(exc)})
            pending.append(source)
            continue
        records.append(record)
    coverage = {"status": "complete" if not pending else "partial", "unique_images": len(inventory),
                "files": sum(len(r["sources"]) for r in inventory), "analyzed": len(records),
                "pending": len(pending), "invalid": invalid,
                "visually_reviewed": sum("visual_review" in r for r in records),
                "corpora": dict(Counter(s["corpus"] for r in inventory for s in r["sources"])),
                "image_kinds": dict(Counter(r["analysis"]["image_kind"] for r in records))}
    return records, pending, coverage


def build_gallery(records, pending, coverage, folder, output):
    thumbs = folder / "thumbnails-srgb-v1"
    thumbs.mkdir(exist_ok=True)
    items = []
    for record in [*records, *pending]:
        item = {k: record.get(k) for k in ("id", "sources", "analysis", "visual_review")}
        if item["analysis"]:
            item["analysis"] = dict(item["analysis"], lessons=workflow_lessons(item["analysis"]))
        thumb = thumbs / (record["id"] + ".jpg")
        if not thumb.exists():
            with Image.open(record["sources"][0]["path"]) as image:
                image = image_to_srgb(ImageOps.exif_transpose(image))
                image.thumbnail((320, 320), Image.Resampling.LANCZOS)
                image.save(thumb, quality=78)
        item["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(thumb.read_bytes()).decode()
        item["source_url"] = Path(record["sources"][0]["path"]).as_uri()
        items.append(item)
    payload = json.dumps({"coverage": coverage, "items": items}, ensure_ascii=False).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><title>哈苏作品逐图学习</title>
<style>body{background:#151515;color:#e8e2d8;font:15px/1.65 system-ui;margin:32px}h1{font-size:28px}a{color:#e4b982}input,select{font:inherit;padding:10px;background:#252525;color:inherit;border:1px solid #666;border-radius:8px}header{position:sticky;top:0;background:#151515;padding:10px 0;z-index:1}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:20px}.card{background:#222;border-radius:12px;overflow:hidden;padding:18px}.card img{width:100%;height:245px;object-fit:contain;background:#191919}.meta{font-size:12px;color:#aaa;overflow-wrap:anywhere}h3{font-size:17px}h4{margin-bottom:5px;color:#e4b982}details{margin-top:12px}li{margin:10px 0}.pending{opacity:.55}.count{color:#e4b982}p{margin:8px 0}</style>
<h1>哈苏作品逐图学习</h1><p>每张图片独立记录可见优点、后期适用条件和拍摄依赖。分析是对成片的视觉判断，无法反推出原片或作者的真实参数。</p>
<header><p id="coverage" class="count"></p><input id="search" placeholder="搜索作者、画面、后期要点" size="32"><select id="kind"><option value="">全部类型</option><option value="photograph">摄影作品</option><option value="behind_scenes">制作花絮</option><option value="portrait_reference">肖像参考</option><option value="layout">书页与拼图</option><option value="pending">尚待分析</option></select><span id="count"></span></header><div class="grid" id="grid"></div>
<script>const data=__DATA__;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const kindNames={photograph:'摄影作品',behind_scenes:'制作花絮',portrait_reference:'肖像参考',layout:'书页与拼图',nonphotographic:'其他图像'};
document.getElementById('coverage').textContent=`本地视觉模型已逐图分析 ${data.coverage.analyzed} / ${data.coverage.unique_images} 张文件；抽检 ${data.coverage.visually_reviewed||0} 张；待完成 ${data.coverage.pending} 张。`;
function render(){const q=document.getElementById('search').value.toLowerCase(),kind=document.getElementById('kind').value;const items=data.items.filter(r=>(!kind||(r.analysis?.image_kind||'pending')===kind)&&JSON.stringify([r.sources,r.analysis]).toLowerCase().includes(q));document.getElementById('count').textContent=` ${items.length} 张`;document.getElementById('grid').innerHTML=items.map(r=>{const a=r.analysis,s=r.sources[0];return `<article class="card ${a?'':'pending'}"><a href="${esc(r.source_url)}"><img loading="lazy" src="${r.thumbnail}"></a><p class="meta">${esc(s.year)} · ${esc(s.photographer)} · ${esc(s.category)}<br>${esc(s.relative_path)} · ${s.role==='jury_or_program'?'评委／项目肖像':esc(kindNames[a?.image_kind]||'待分析')}</p><h3>${esc(a?.caption||'尚待逐图分析')}</h3>${a?`<details><summary>好在哪里 · 后期指导</summary><h4>可见优点</h4><ul>${a.strengths.map(x=>`<li>${esc(x.evidence)}<br><strong>作用：</strong>${esc(x.benefit)}</li>`).join('')}</ul><h4>后期如何借鉴</h4>${a.lessons.map(x=>`<p><strong>适用：</strong>${esc(x.applies_when)}<br><strong>操作：</strong>${esc(x.action)}<br><strong>避免：</strong>${esc(x.avoid)}<br><strong>复核：</strong>${esc(x.review)}</p>`).join('')}<h4>依赖拍摄现场</h4><ul>${a.capture_dependencies.map(x=>`<li>${esc(x)}</li>`).join('')}</ul><p class="meta">${esc(a.uncertainty)}</p></details>`:''}<p class="meta">${esc(r.id)}</p></article>`}).join('')}
document.getElementById('search').oninput=render;document.getElementById('kind').onchange=render;render();</script>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace("__DATA__", payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/references/hasselblad-study"))
    parser.add_argument("--out", type=Path, default=Path(".task-cache/hasselblad-study.html"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    records, pending, coverage = collect(args.input)
    write_json(args.input / "coverage.json", coverage)
    print(json.dumps(coverage, ensure_ascii=False), flush=True)
    if pending and not args.allow_partial:
        raise SystemExit("Refusing to publish an incomplete study index")
    if args.bundle:
        if pending:
            raise SystemExit("Runtime bundle requires complete image coverage")
        compact = [{k: r[k] for k in ("id", "sha256", "schema_version", "status", "sources", "features", "analysis", "model", "analyzed_at", "visual_review") if k in r}
                   for r in records]
        write_json(args.bundle, {"version": 1, "built_at": datetime.now(timezone.utc).isoformat(), "coverage": coverage, "records": compact})
    build_gallery(records, pending, coverage, args.input, args.out)
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
