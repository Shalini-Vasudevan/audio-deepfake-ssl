"""Tiny synthetic stand-in for LA + ITW so the pipeline can be smoke-tested offline."""
import os, sys, numpy as np, soundfile as sf
root = sys.argv[1]; rng = np.random.default_rng(0)
def wav(bona):
    t = np.arange(16000*3)/16000; f = rng.uniform(100,250)
    x = np.sin(2*np.pi*f*t)*(0.5+0.5*np.sin(2*np.pi*3*t))
    return (x + (0.02 if bona else 0.2)*rng.standard_normal(len(t))).astype(np.float32)
la = f"{root}/LA"; os.makedirs(f"{la}/ASVspoof2019_LA_cm_protocols", exist_ok=True)
for part, n, off in [("train",40,0),("dev",16,100),("eval",24,200)]:
    os.makedirs(f"{la}/ASVspoof2019_LA_{part}/flac", exist_ok=True); lines=[]
    for i in range(n):
        b = i%4==0; u=f"LA_{'T' if part=='train' else 'D' if part=='dev' else 'E'}_{off+i:07d}"
        sf.write(f"{la}/ASVspoof2019_LA_{part}/flac/{u}.flac", wav(b), 16000)
        lines.append(f"LA_{off+i%5:04d} {u} - {'-' if b else 'A07'} {'bonafide' if b else 'spoof'}")
    nm={"train":"train.trn","dev":"dev.trl","eval":"eval.trl"}[part]
    open(f"{la}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{nm}.txt","w").write("\n".join(lines))
# fix speaker disjointness: speakers offset by part already (off + i%5)
os.makedirs(f"{la}/ASVspoof2019_LA_asv_scores", exist_ok=True)
sc=[f"LA_0001 target {rng.normal(3,1):.3f}" for _ in range(50)]+[f"LA_0001 nontarget {rng.normal(-3,1):.3f}" for _ in range(50)]+[f"LA_0001 spoof {rng.normal(1,1.5):.3f}" for _ in range(50)]
open(f"{la}/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt","w").write("\n".join(sc))
w=f"{root}/itw/release_in_the_wild"; os.makedirs(w,exist_ok=True); rows=["file,speaker,label"]
for i in range(20):
    b=i%2==0; sf.write(f"{w}/{i}.wav", wav(b), 16000); rows.append(f"{i}.wav,s{i},{'bona-fide' if b else 'spoof'}")
open(f"{w}/meta.csv","w").write("\n".join(rows))
