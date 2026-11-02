"""Build a standalone replay of saved poses (no simulated map evolution)."""
import argparse
import json
from pathlib import Path


def build(root: Path, output: Path):
    manifest = json.loads((root / "data/heldout/manifest.json").read_text(encoding="utf-8"))
    cases = {c["name"]: c for c in manifest["cases"]}
    examples = json.loads((root / "data/traces/selected_examples.json").read_text(encoding="utf-8"))["demos"]
    for demo in examples:
        demo["case"] = cases[demo["scene"]]
        for key in ("cnn_motion", "fixed_motion"):
            trace = json.loads((root / demo[key]).read_text(encoding="utf-8"))
            # Render every original saved pose; retain its simulated timestamp.
            demo[key] = [[p["time_s"], p["x"], p["y"], p["yaw_rad"]] for p in trace]
    document = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><title>OBYM saved-route replay</title>
<style>body{font:16px system-ui;margin:24px;background:#f8fafc;color:#172033}main{max-width:1400px;margin:auto}canvas{width:100%;background:white;border:1px solid #cbd5e1}input{width:55%}button,select{padding:8px}p{line-height:1.5}.cnn{color:#2563eb}.fixed{color:#b45309}</style>
<main><h1>Observe Before You Move: saved-route replay</h1>
<p>One median CNN gain and one median CNN loss among differing decisions. Real archived poses and matching obstacles; this replay contains no map evolution or new simulation. Positive gain = FixedMinus45 time − ValueCNN time.</p>
<p><button id="play">Play</button> <input aria-label="Simulated time" id="seek" type="range" min="0" max="50" step="0.01" value="0"> <output id="time"></output> <select id="speed" aria-label="Playback speed"><option value="1">1×</option><option value="4" selected>4×</option></select></p>
<canvas id="canvas" width="1400" height="670" aria-label="Side by side routes with robot poses"></canvas>
<p><span class="cnn">Blue: ValueCNN</span> · <span class="fixed">Orange dashed: FixedMinus45</span> · Green circle: start · Red cross: goal · Star: optional-look decision. The arrow shows recorded heading. Stationary sensing and turning consume simulated time; each robot stops at its saved final pose.</p></main>
<script>
const demos=__DATA__;
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),seek=document.getElementById('seek'),time=document.getElementById('time'),play=document.getElementById('play');
const maxT=Math.max(...demos.flatMap(d=>[d.cnn_time_s,d.fixed_time_s]));seek.max=maxT;
function point(x,y,left){return [left+65+(x+5)*57,580-(y+4)*57]}
function line(x1,y1,x2,y2,color,w=2){ctx.strokeStyle=color;ctx.lineWidth=w;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke()}
function star(x,y,color){ctx.fillStyle=color;ctx.beginPath();for(let k=0;k<10;k++){let a=-Math.PI/2+k*Math.PI/5,r=k%2?4:10;let px=x+r*Math.cos(a),py=y+r*Math.sin(a);k?ctx.lineTo(px,py):ctx.moveTo(px,py)}ctx.closePath();ctx.fill()}
function draw(){ctx.clearRect(0,0,1400,670);let t=+seek.value;time.textContent=t.toFixed(2)+' simulated s';demos.forEach((d,j)=>{let left=j*700;ctx.fillStyle='#172033';ctx.font='bold 18px system-ui';ctx.fillText((j?'CNN loss':'CNN gain')+' | '+d.scene,left+30,30);ctx.font='16px system-ui';ctx.fillText('Fixed − CNN = '+(d.gain_s>0?'+':'')+d.gain_s.toFixed(3)+' s',left+30,56);
for(let x=-5;x<=5;x++){let [px,py]=point(x,-4,left);line(px,py,px,point(x,4,left)[1],'#e2e8f0',1);ctx.fillStyle='#64748b';ctx.fillText(x,px-4,603)}for(let y=-4;y<=4;y++){let [px,py]=point(-5,y,left);line(px,py,point(5,y,left)[0],py,'#e2e8f0',1);ctx.fillStyle='#64748b';ctx.fillText(y,px-25,py+5)}ctx.fillText('x (m)',left+330,630);
ctx.fillStyle='#64748b';for(let [size,pos] of d.case.geometry){let [x,y]=point(pos[0]-size[0]/2,pos[1]+size[1]/2,left);ctx.fillRect(x,y,size[0]*57,size[1]*57)}
for(let [key,color,label,choice,duration] of [['cnn_motion','#2563eb','ValueCNN',d.cnn_choice,d.cnn_time_s],['fixed_motion','#d97706','FixedMinus45',d.fixed_choice,d.fixed_time_s]]){let rows=d[key],k=0;while(k+1<rows.length&&rows[k+1][0]<=t)k++;let p=rows[k];ctx.setLineDash(key==='fixed_motion'?[7,5]:[]);ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();for(let i=0;i<=k;i++){let [x,y]=point(rows[i][1],rows[i][2],left);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.stroke();ctx.setLineDash([]);let [x,y]=point(p[1],p[2],left);ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,6,0,2*Math.PI);ctx.fill();line(x,y,x+18*Math.cos(p[3]),y-18*Math.sin(p[3]),color,3);if(choice&&t>=choice.sim_time_s){let q=point(...choice.xy,left);star(q[0],q[1],color)}ctx.fillStyle=color;ctx.font='15px system-ui';ctx.fillText(label+': '+duration.toFixed(3)+' s; '+(choice?'look '+choice.selected_offset_deg+'°':'skip optional look'),left+30,key==='cnn_motion'?82:105)}
let [sx,sy]=point(...d.case.start_xy,left);ctx.strokeStyle='#16a34a';ctx.lineWidth=3;ctx.beginPath();ctx.arc(sx,sy,9,0,Math.PI*2);ctx.stroke();let [gx,gy]=point(...d.case.goal_xy,left);line(gx-7,gy-7,gx+7,gy+7,'#dc2626',4);line(gx-7,gy+7,gx+7,gy-7,'#dc2626',4)
})}
let playing=false,last;play.onclick=()=>{playing=!playing;if(playing&&+seek.value>=maxT)seek.value=0;play.textContent=playing?'Pause':'Play';last=undefined};seek.oninput=draw;
function tick(now){if(playing&&last!==undefined){seek.value=Math.min(maxT,+seek.value+(now-last)/1000*+document.getElementById('speed').value);draw();if(+seek.value>=maxT){playing=false;play.textContent='Play'}}last=now;requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);
</script></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document.replace("__DATA__", json.dumps(examples, separators=(",", ":"))), encoding="utf-8")
    print("wrote", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/generated/replay.html"))
    args = parser.parse_args(); build(args.root.resolve(), args.output.resolve())
