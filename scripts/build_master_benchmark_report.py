"""Build an offline, unaltered-image report from actual benchmark exports.

Usage:
    .venv/bin/python scripts/build_master_benchmark_report.py MANIFEST.json
    .venv/bin/python scripts/build_master_benchmark_report.py MANIFEST.json --out REPORT.html

See master_benchmark_manifest.example.json for the manifest shape. Relative
asset paths resolve against the manifest directory, or its optional base_dir.
Baseline/improved files must have status="exported". This is the manifest
author's assertion, not proof of the rendering engine: record engine and an
optional evidence_path to its export/audit JSON. Never mark an intermediate
or approximate preview as a native application export. Missing stages fail by
default; --allow-partial creates a prominently marked draft for inspection.
RAW/TIFF sources need an existing browser-readable preview_path. The script
reads metadata and embeds original image bytes; it never renders, crops,
resizes, color-converts, or otherwise edits an image. roi=[left,top,right,bottom]
only changes CSS display, in the stored image's pixel coordinates.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


STAGES = ("original", "baseline", "improved", "reference")
LABELS = {"original": "原片 / 练习起点", "baseline": "改善前实际导出",
          "improved": "改善后实际导出", "reference": "摄影师参考成片"}
MIMES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
         "AVIF": "image/avif"}
DEFAULT_OUT = Path(".analysis/master-benchmark-2026-09-19/index.html")


def strings(value: object, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError(f"{where} must be a list of strings")
    return value


def source_link(value: object, where: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object with label and url")
    url = str(value.get("url", ""))
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError(f"{where} requires an http(s) source URL")
    return {"label": str(value.get("label", "来源")), "url": url}


def asset_path(value: object, base: Path, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} requires a nonempty local file path")
    path = Path(value).expanduser()
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise ValueError(f"{where}: actual file does not exist: {path}")
    return path


def load_photo(value: object, stage: str, base: Path, assets: dict,
               where: str) -> dict | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = {"path": value}
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an image object or null")
    if not value.get("path"):
        return None
    if stage in {"baseline", "improved"} and value.get("status") != "exported":
        raise ValueError(f'{where} requires status="exported" for an actual export')
    path = asset_path(value["path"], base, f"{where}.path")
    preview = asset_path(value.get("preview_path", value["path"]), base,
                         f"{where}.preview_path")
    evidence = (asset_path(value["evidence_path"], base, f"{where}.evidence_path")
                if value.get("evidence_path") else None)
    blob = preview.read_bytes()
    try:
        with Image.open(preview) as im:
            width, height = im.size
            fmt = im.format
            orientation = int(im.getexif().get(274, 1))
            icc = bool(im.info.get("icc_profile"))
        with Image.open(preview) as im:
            im.verify()
    except Exception as exc:
        raise ValueError(f"{where}: cannot read display image; RAW/TIFF needs an "
                         f"existing JPEG/PNG/WebP/AVIF preview_path: {preview}") from exc
    if fmt not in MIMES:
        raise ValueError(f"{where}: {fmt} is not an accepted browser image; "
                         "provide a browser-readable preview_path")
    roi = value.get("roi")
    if roi is not None:
        if (not isinstance(roi, list) or len(roi) != 4
                or any(isinstance(n, bool) or not isinstance(n, (int, float))
                       or not math.isfinite(n) for n in roi)):
            raise ValueError(f"{where}.roi must be [left, top, right, bottom]")
        left, top, right, bottom = roi
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"{where}.roi lies outside the {width} x {height} image")
        if orientation != 1:
            raise ValueError(f"{where}.roi needs a display image with EXIF orientation 1")
        display_width, display_height = right - left, bottom - top
    else:
        display_width, display_height = ((height, width) if orientation in {5, 6, 7, 8}
                                         else (width, height))
    digest = hashlib.sha256(blob).hexdigest()
    if digest not in assets:
        assets[digest] = {
            "data_url": f"data:{MIMES[fmt]};base64," + base64.b64encode(blob).decode(),
            "width": width, "height": height, "bytes": len(blob), "format": fmt,
            "icc_profile_present": icc,
        }
    return {
        "stage": stage, "label": str(value.get("label", LABELS[stage])),
        "kind": str(value.get("kind", "现有图像文件")), "asset": digest,
        "engine": str(value.get("engine", "未声明")),
        "evidence_path": str(evidence) if evidence else None,
        "path": str(path), "preview_path": str(preview), "filename": preview.name,
        "source_sha256": (digest if path == preview else
                          hashlib.sha256(path.read_bytes()).hexdigest()),
        "preview_is_separate": path != preview, "roi": roi,
        "display_width": display_width, "display_height": display_height,
        "notes": strings(value.get("notes"), f"{where}.notes"),
        "source_url": (source_link({"url": value["source_url"]}, where)["url"]
                       if value.get("source_url") else None),
    }


def prepare_report(manifest: Path, allow_partial: bool = False) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("manifest requires a nonempty cases list")
    base = Path(data.get("base_dir", ".")).expanduser()
    base = (manifest.parent / base).resolve() if not base.is_absolute() else base.resolve()
    assets, cases, ids, missing = {}, [], set(), []
    for number, item in enumerate(data["cases"], 1):
        where = f"case {number}"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")
        case_id = str(item.get("id", f"case-{number}"))
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        inputs = item.get("images", item)
        if not isinstance(inputs, dict):
            raise ValueError(f"{where}.images must be an object")
        images = {stage: load_photo(inputs.get(stage), stage, base, assets,
                                   f"{where}.{stage}") for stage in STAGES}
        absent = [LABELS[stage] for stage, photo in images.items() if photo is None]
        if absent:
            missing.append(f"{case_id}: {', '.join(absent)}")
        limitations = strings(item.get("limitations"), f"{where}.limitations").copy()
        limitations.insert(0, "图像几何未配准，可能存在裁切、透视、镜头校正或尺寸差异；"
                            "以区域明暗、颜色关系和主体层级为比较目标。")
        if images["baseline"] and images["improved"]:
            if images["baseline"]["asset"] == images["improved"]["asset"]:
                limitations.append("改善前后预览的文件字节完全相同；本报告不能据此声称视觉改善。")
        links = [source_link(link, f"{where}.source_links")
                 for link in item.get("source_links", [])]
        if item.get("source_page"):
            links.insert(0, source_link({"label": "摄影师原文 / 官方教程",
                                        "url": item["source_page"]}, where))
        metrics = item.get("metrics", [])
        if not isinstance(metrics, list) or any(not isinstance(row, dict) for row in metrics):
            raise ValueError(f"{where}.metrics must be a list of objects")
        rows = []
        for row in metrics:
            if not row.get("label"):
                raise ValueError(f"{where}: every metric requires a label")
            clean = {}
            for key in ("label", *STAGES, "unit", "note"):
                value = row.get(key)
                if value is not None and not isinstance(value, (str, int, float)):
                    raise ValueError(f"{where}.metrics.{key} must be text or a number")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{where}.metrics.{key} cannot be NaN or infinity")
                clean[key] = None if value is None else str(value)
            rows.append(clean)
        cases.append({
            "id": case_id, "title": str(item.get("title", case_id)),
            "author": str(item.get("author", "")),
            "description": str(item.get("description", "")),
            "images": images, "missing": absent, "source_links": links,
            "observations": strings(item.get("observations"), f"{where}.observations"),
            "metrics": rows, "limitations": limitations,
        })
    if missing and not allow_partial:
        raise ValueError("Refusing to build a final report without actual files for all stages:\n  "
                         + "\n  ".join(missing)
                         + "\nUse --allow-partial only for a clearly marked draft.")
    return {
        "title": str(data.get("title", "摄影师案例 · SuperPhoto 实测对照")),
        "subtitle": str(data.get("subtitle", "比较实际修图结果中的光影、颜色与视觉层级。")),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest.resolve()), "incomplete": bool(missing),
        "summary": strings(data.get("summary"), "summary"),
        "cases": cases, "assets": assets,
    }


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>摄影师案例 · 实测对照</title>
<style>
:root{color-scheme:dark;--bg:#151719;--panel:#1c1f22;--line:#34393e;--ink:#f0efeb;--muted:#adb4b9;--accent:#dfb881;--draft:#f2c567}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1520px;margin:auto;padding:38px 24px 72px}h1{font-size:clamp(24px,3vw,36px);line-height:1.3;margin:4px 0 12px}h2{font-size:24px;margin:0}h3{font-size:16px;margin:22px 0 8px}p{margin:8px 0}a{color:var(--accent);text-underline-offset:3px}button,select{font:inherit;color:inherit;background:#24292d;border:1px solid #48515a;border-radius:7px;padding:7px 12px;cursor:pointer}button:hover,select:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:3px}button.active{background:#e0be90;color:#201b14;border-color:#e0be90}button:disabled{opacity:.4;cursor:not-allowed}.muted,.small{color:var(--muted)}.small{font-size:12px}.eyebrow{color:var(--accent);font-size:12px;letter-spacing:.12em}.intro{max-width:920px}.summary{padding-left:20px}.summary li{margin:7px 0}.notice{border:1px solid #a67b2c;background:#30291d;border-radius:10px;padding:15px 18px;color:var(--draft);margin:22px 0}.jump{display:flex;flex-wrap:wrap;gap:10px 20px;margin:25px 0}.case{margin:38px 0 0;padding:26px;background:var(--panel);border:1px solid var(--line);border-radius:14px;scroll-margin-top:20px}.case-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px 28px;align-items:start}.source-links{display:flex;gap:10px 18px;flex-wrap:wrap;font-size:13px;margin:10px 0}.toolbar,.stage-tabs{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 12px}.views{display:grid;gap:18px;align-items:start}.views.side{grid-template-columns:repeat(2,minmax(0,1fr))}.views.overview{grid-template-columns:repeat(4,minmax(0,1fr))}.views.single{grid-template-columns:1fr;max-width:1160px;margin:auto}.photo{margin:0;min-width:0}.photo-top{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:9px;min-height:38px}.photo-top select{max-width:100%}.stage{aspect-ratio:3/2;background:#101112;border:1px solid #3a4147;display:flex;align-items:center;justify-content:center;overflow:hidden}.photo-frame{position:relative;overflow:hidden;flex:none;max-width:100%;max-height:100%}.photo-frame img{display:block}.empty{color:#b9a883;padding:30px;text-align:center}.photo-caption{padding:10px 0}.photo-caption p{margin:4px 0}.photo-caption .small{overflow-wrap:anywhere}.file-actions{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;margin-top:9px}.text-button{background:transparent;padding:3px 8px;font-size:12px}.tag{font-size:11px;background:#30353a;border:1px solid #444b51;border-radius:4px;padding:2px 6px;color:#c7cdd2}.columns{display:grid;grid-template-columns:1fr 1fr;gap:30px;margin-top:8px}.observations,.limits{margin:0;padding-left:20px}.observations li,.limits li{margin:7px 0}.limits{color:var(--muted);font-size:13px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;margin-top:8px}table{width:100%;border-collapse:collapse;font-size:13px;text-align:left;min-width:690px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#272c30;font-weight:500}tr:last-child td{border-bottom:0}.metric-note{display:block;color:var(--muted);font-size:11px;margin-top:3px}details{margin-top:18px}summary{cursor:pointer;color:var(--muted);font-size:13px}.audit{white-space:pre-wrap;word-break:break-all;font-size:11px;color:var(--muted);padding:12px;background:#15181b;border-radius:6px;max-height:250px;overflow:auto}footer{margin-top:30px;font-size:12px;color:var(--muted)}dialog{background:#171a1d;color:var(--ink);border:1px solid #59616a;border-radius:10px;width:min(96vw,1800px);max-height:96vh;padding:16px}dialog::backdrop{background:#000b}.dialog-bar{display:flex;gap:16px;justify-content:space-between;align-items:center;margin-bottom:12px}.full-image{display:block;max-width:100%;max-height:80vh;object-fit:contain;margin:auto}.roi-note{color:var(--accent);font-size:12px}
@media(max-width:1100px){.views.overview{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:720px){main{padding:24px 12px 50px}.case{padding:17px 12px}.views.side,.views.overview,.columns{grid-template-columns:1fr}.case-head h2{font-size:21px}.toolbar button{padding:7px 9px;font-size:13px}}
</style></head><body><main><div class="eyebrow">SUPERPHOTO / PHOTOGRAPHER BENCHMARK</div><h1 id="title"></h1><p class="intro muted" id="subtitle"></p><div id="status"></div><ul class="summary intro" id="summary"></ul><nav class="jump" id="jump" aria-label="案例跳转"></nav><div id="cases"></div><footer id="footer"></footer></main>
<dialog id="full-view"><div class="dialog-bar"><span id="full-title"></span><button id="close-full" aria-label="关闭完整图片">关闭</button></div><img class="full-image" id="full-image" alt=""><p class="small" id="full-note"></p></dialog>
<script id="report-data" type="application/json">__REPORT_JSON__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById('report-data').textContent);
const stages=['original','baseline','improved','reference'];
const names={original:'原片 / 练习起点',baseline:'改善前实际导出',improved:'改善后实际导出',reference:'摄影师参考成片'};
const $=id=>document.getElementById(id);
const el=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e};
const addLink=(parent,label,url)=>{const a=el('a','',label);a.href=url;a.target='_blank';a.rel='noopener noreferrer';parent.append(a)};
const openFull=photo=>{const a=data.assets[photo.asset];$('full-image').src=a.data_url;$('full-image').alt=photo.label;$('full-title').textContent=photo.label+' · '+photo.filename;$('full-note').textContent=`完整文件预览 ${a.width} × ${a.height}；${photo.roi?'此处显示含应用界面的完整原截图。':'保留原文件图像内容。'}`;$('full-view').showModal()};
$('close-full').onclick=()=>$('full-view').close();
$('full-view').addEventListener('click',event=>{if(event.target===$('full-view')){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close()}});
document.title=data.title;$('title').textContent=data.title;$('subtitle').textContent=data.subtitle;
if(data.incomplete){$('status').append(el('div','notice','未完成审阅稿：部分阶段尚无实际导出。空位表示没有结果，不能据此声称已完成改善或验证。'))}
for(const line of data.summary)$('summary').append(el('li','',line));
function picture(c,stage,selectable,onSelect){
  const photo=c.images[stage],fig=el('figure','photo'),top=el('div','photo-top');
  if(selectable){const select=el('select');select.setAttribute('aria-label','选择对照阶段');for(const key of stages){const o=el('option','',names[key]+(c.images[key]?'':' · 暂无结果'));o.value=key;select.append(o)}select.value=stage;select.onchange=()=>onSelect(select.value);top.append(select)}
  else top.append(el('strong','',photo?.label||names[stage]));
  if(photo)top.append(el('span','tag',photo.kind));fig.append(top);
  const view=el('div','stage');fig.append(view);
  if(!photo){view.append(el('p','empty',names[stage]+'\n尚无实际文件'));return fig}
  const asset=data.assets[photo.asset],ratio=photo.display_width/photo.display_height;
  const frame=el('div','photo-frame');frame.style.aspectRatio=String(ratio);frame.style.width=ratio>=1.5?'100%':'auto';frame.style.height=ratio>=1.5?'auto':'100%';
  const img=el('img');img.src=asset.data_url;img.alt=c.title+' — '+photo.label;img.loading='lazy';
  if(photo.roi){const [l,t,r,b]=photo.roi,w=r-l,h=b-t;Object.assign(img.style,{position:'absolute',width:(asset.width/w*100)+'%',height:(asset.height/h*100)+'%',maxWidth:'none',left:(-l/w*100)+'%',top:(-t/h*100)+'%'})}
  else Object.assign(img.style,{width:'100%',height:'100%',objectFit:'contain'});
  frame.append(img);view.append(frame);
  const cap=el('figcaption','photo-caption');cap.append(el('p','small',`${photo.label} · 显示区域 ${photo.display_width} × ${photo.display_height}`));
  if(stage==='baseline'||stage==='improved')cap.append(el('p','small','清单声明的渲染引擎：'+photo.engine));
  if(photo.roi)cap.append(el('p','roi-note','教程截图，仅显示核验过的照片区域；完整截图可查看。'));
  if(photo.preview_is_separate)cap.append(el('p','small','RAW / 原始文件使用已存在的预览显示；预览不等于未经处理的传感器数据。'));
  for(const note of photo.notes)cap.append(el('p','small',note));
  const actions=el('div','file-actions'),full=el('button','text-button',photo.roi?'查看完整截图':'查看完整预览');full.onclick=()=>openFull(photo);actions.append(full);
  const download=el('a','small','下载原预览文件');download.href=asset.data_url;download.download=photo.filename;actions.append(download);
  if(photo.source_url)addLink(actions,'图像来源',photo.source_url);cap.append(actions);fig.append(cap);return fig;
}
data.cases.forEach((c,index)=>{
  const section=el('section','case');section.id='case-'+index;
  const jump=el('a','',c.title);jump.href='#'+section.id;$('jump').append(jump);
  const head=el('div','case-head'),headtext=el('div');headtext.append(el('div','eyebrow',String(index+1).padStart(2,'0')+' / '+c.author),el('h2','',c.title));head.append(headtext);section.append(head);
  if(c.description)section.append(el('p','muted',c.description));
  const links=el('div','source-links');for(const link of c.source_links)addLink(links,link.label,link.url);section.append(links);
  if(c.missing.length)section.append(el('p','notice','本案例缺少：'+c.missing.join('、')+'。'));
  const toolbar=el('div','toolbar'),tabs=el('div','stage-tabs'),views=el('div','views side');section.append(toolbar,tabs,views);
  let mode='side',left=c.images.baseline?'baseline':'original',right=c.images.improved?'improved':'reference',single=c.images.improved?'improved':c.images.baseline?'baseline':'original';
  const modes={side:'并排对照',single:'同位置切换',overview:'四图总览'},buttons={};
  function render(){
    views.replaceChildren();views.className='views '+mode;tabs.replaceChildren();
    for(const key of Object.keys(modes)){buttons[key].classList.toggle('active',mode===key);buttons[key].setAttribute('aria-pressed',String(mode===key))}
    if(mode==='side'){views.append(picture(c,left,true,key=>{left=key;render()}),picture(c,right,true,key=>{right=key;render()}))}
    else if(mode==='overview'){for(const key of stages)views.append(picture(c,key,false))}
    else{for(const key of stages){const b=el('button',single===key?'active':'',names[key]);b.setAttribute('aria-pressed',String(single===key));b.onclick=()=>{single=key;render()};tabs.append(b)}views.append(picture(c,single,false))}
  }
  for(const key of Object.keys(modes)){const b=el('button','',modes[key]);buttons[key]=b;b.onclick=()=>{mode=key;render()};toolbar.append(b)}render();
  const cols=el('div','columns'),observations=el('div'),limits=el('div');observations.append(el('h3','','观察记录'));
  if(c.observations.length){const list=el('ul','observations');for(const text of c.observations)list.append(el('li','',text));observations.append(list)}else observations.append(el('p','muted','尚未填写观察；本页面不会自动推断改善结论。'));
  limits.append(el('h3','','比较边界'));const limitList=el('ul','limits');for(const text of c.limitations)limitList.append(el('li','',text));limits.append(limitList);cols.append(observations,limits);section.append(cols);
  if(c.metrics.length){section.append(el('h3','','实验记录中的数值'),el('p','small','数值由实验清单提供；“—”表示未提供。本页面不把不同图幅的像素差异解释为质量分数。'));const wrap=el('div','table-wrap'),table=el('table'),thead=el('thead'),tr=el('tr');for(const text of ['观察项',...stages.map(s=>names[s])])tr.append(el('th','',text));thead.append(tr);table.append(thead);const tbody=el('tbody');for(const row of c.metrics){const r=el('tr'),label=el('td','',row.label+(row.unit?' ('+row.unit+')':''));if(row.note)label.append(el('span','metric-note',row.note));r.append(label);for(const key of stages)r.append(el('td','',row[key]??'—'));tbody.append(r)}table.append(tbody);wrap.append(table);section.append(wrap)}
  const audit=el('details');audit.append(el('summary','','文件路径、显示范围与校验值'));const lines=[];for(const key of stages){const p=c.images[key];if(!p){lines.push(names[key]+': missing');continue}lines.push(names[key]+': '+p.path,'预览: '+p.preview_path,'清单声明的渲染引擎: '+p.engine,'导出证据文件: '+(p.evidence_path||'未提供'),'预览 SHA-256: '+p.asset,'素材 SHA-256: '+p.source_sha256,'CSS ROI: '+JSON.stringify(p.roi),p.preview_is_separate?'使用独立预览':'直接嵌入文件原字节','')}audit.append(el('div','audit',lines.join('\n')));section.append(audit);$('cases').append(section);
});
$('footer').textContent=`${data.incomplete?'未完成审阅稿':'四阶段实际文件齐备'} · ${data.cases.length} 个案例 · 图片原字节内嵌，未生成或修改照片。构建时间 ${data.built_at}。素材版权归各摄影师；本地学习对照，不代表获得产品分发许可。`;
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="JSON case manifest")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Build a visibly incomplete draft; never invent missing stages")
    args = parser.parse_args()
    try:
        report = prepare_report(args.manifest.resolve(), args.allow_partial)
        output = args.out.resolve()
        inputs = {Path(p[field]) for c in report["cases"] for p in c["images"].values()
                  if p for field in ("path", "preview_path")}
        inputs.update(Path(p["evidence_path"]) for c in report["cases"]
                      for p in c["images"].values() if p and p["evidence_path"])
        if output == args.manifest.resolve() or output in inputs:
            raise ValueError("output must not overwrite the manifest, source images, or evidence")
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(HTML.replace("__REPORT_JSON__", payload), encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps({"output": str(output), "cases": len(report["cases"]),
                      "status": "incomplete_draft" if report["incomplete"] else "complete_files",
                      "embedded_images": len(report["assets"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
