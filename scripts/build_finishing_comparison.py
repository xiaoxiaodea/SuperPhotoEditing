"""Package a completed job's actual exports in a standalone comparison page."""
import argparse
import base64
import json
from pathlib import Path
import sqlite3
from uuid import UUID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=UUID)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    with sqlite3.connect(f"file:{data / 'superphoto.db'}?mode=ro", uri=True) as conn:
        job = conn.execute("SELECT status FROM jobs WHERE id = ?", (str(args.job_id),)).fetchone()
        if not job or job[0] != "ready":
            raise ValueError("Comparison requires a completed job")
        titles = dict(conn.execute("SELECT slot, title FROM variants WHERE job_id = ?", (str(args.job_id),)))
    root = data / "jobs" / str(args.job_id)
    def uri(path):
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    photos = {"original": {"title": "原片", "src": uri(root / "lightroom-exports/baseline/baseline.jpg")}}
    for slot in "ABC":
        if slot not in titles:
            continue
        audit = json.loads((root / f"lightroom-audits/variant_{slot}.json").read_text(encoding="utf-8"))
        exported = root / f"variant_{slot}.jpg"
        if exported.read_bytes() != Path(audit["accepted_export"]).read_bytes():
            raise ValueError(f"{slot} does not match its accepted Lightroom export")
        photos[slot] = {"title": f"{slot} · {titles[slot]}", "src": uri(exported)}
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SuperPhoto · 后期对比</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#191919;color:#eee;font:16px/1.6 system-ui,sans-serif}
main{max-width:1600px;margin:auto;padding:28px}h1{font-size:26px;margin:0}p{color:#bbb}
nav{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}button,a{font:inherit;color:inherit}
button{cursor:pointer;background:#292929;border:1px solid #555;border-radius:8px;padding:10px 18px}
button[aria-pressed=true]{background:#e1d4b5;color:#161616;border-color:#e1d4b5}
.viewer{position:relative;max-height:72vh;width:fit-content;max-width:100%;margin:auto}
.viewer img{display:block;max-height:72vh;max-width:100%;width:auto;user-select:none}
#before{position:absolute;inset:0;height:100%;width:100%;clip-path:inset(0 50% 0 0)}
#line{position:absolute;left:50%;top:0;bottom:0;width:2px;background:#eee;pointer-events:none}
.labels{display:flex;justify-content:space-between;margin:16px auto 6px;max-width:700px}
.slider{display:block;width:100%;max-width:700px;margin:0 auto 28px;accent-color:#d8c9a4}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-top:28px}
figure{margin:0}figure img{width:100%;display:block}figcaption{padding:10px 0}a{display:inline-block;margin:6px 0}
@media(max-width:700px){main{padding:16px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><main><h1>这张照片的后期对比</h1>
<p>点击方案，拖动滑块对比原片。数字 1 / 2 / 3 可以切换方案。</p>
<nav id="choices" aria-label="后期方案"></nav>
<div class="viewer"><img id="after" alt="后期照片"><img id="before" alt="原片"><div id="line"></div></div>
<div class="labels"><span>原片</span><span id="selected"></span></div>
<input id="split" class="slider" type="range" min="0" max="100" value="50" aria-label="原片对比位置">
<div class="grid" id="grid"></div></main>
<script id="photos" type="application/json">__PHOTOS__</script>
<script>
const photos=JSON.parse(document.getElementById('photos').textContent);
const get=id=>document.getElementById(id);
function choose(slot){if(!photos[slot])return;get('after').src=photos[slot].src;get('selected').textContent=photos[slot].title;
 document.querySelectorAll('nav button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.slot===slot)));}
for(const [slot,photo] of Object.entries(photos)){
 if(slot!=='original'){const b=document.createElement('button');b.textContent=photo.title;b.dataset.slot=slot;
 b.onclick=()=>choose(slot);get('choices').append(b);}
 const f=document.createElement('figure'),img=document.createElement('img'),cap=document.createElement('figcaption'),link=document.createElement('a');
 img.src=photo.src;img.alt=photo.title;cap.textContent=photo.title;link.textContent='保存照片';link.href=photo.src;link.download=slot+'.jpg';
 f.append(img,cap,link);get('grid').append(f);
}
get('before').src=photos.original.src;choose(Object.keys(photos).find(slot=>slot!=='original'));
get('split').oninput=e=>{get('before').style.clipPath=`inset(0 ${100-e.target.value}% 0 0)`;get('line').style.left=e.target.value+'%';};
document.addEventListener('keydown',e=>{if(['1','2','3'].includes(e.key))choose('ABC'[Number(e.key)-1]);});
</script></html>'''
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html.replace("__PHOTOS__", json.dumps(photos, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
