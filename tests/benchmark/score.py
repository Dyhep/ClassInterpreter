import sys, json, re, time, itertools
sys.path.insert(0, "/home/claude/ClassInterpreter/app")
import os; os.environ.setdefault("CI_MODELS_DIR", "/home/claude/models")
import numpy as np, soundfile as sf, asr
NUM={"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine","10":"ten"}
def norm(t):
    t=t.lower().replace("%"," percent").replace("-"," ")
    t=re.sub(r"\b(\d+)\b", lambda m: NUM.get(m.group(1), m.group(1)), t)
    t=re.sub(r"[^a-z' ]"," ",t); t=t.replace("'","")
    return t.split()
def align(r, h):
    n,m=len(r),len(h); d=np.zeros((n+1,m+1),int); d[:,0]=range(n+1); d[0,:]=range(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i,j]=min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+(r[i-1]!=h[j-1]))
    i,j=n,m; S=D=I=0
    while i>0 or j>0:
        if i>0 and j>0 and d[i,j]==d[i-1,j-1]+(r[i-1]!=h[j-1]): S+=r[i-1]!=h[j-1]; i-=1; j-=1
        elif i>0 and d[i,j]==d[i-1,j]+1: D+=1; i-=1
        else: I+=1; j-=1
    return S,D,I
MODELS={}
def run(wav, cfg, model):
    asr.CFG.update(cfg)
    if model not in MODELS: MODELS[model]=asr.make_recognizer(model, 4)
    a,_=sf.read(wav, dtype="float32"); finals=[]
    r=asr.Recognizer(lambda: MODELS[model], lambda t: None, lambda t,s: finals.append(t), live_preview=False)
    for i in range(0,len(a),1600): r.push(a[i:i+1600])
    r.push(np.zeros(32000,np.float32)); r.stop()
    return finals
if __name__=="__main__":
    d=sys.argv[1]; ref=norm(" ".join(json.load(open(d+"/ref.json"))))
    BASE=dict(max_seg=7.0,min_silence=0.3,threshold=0.5,pad_pre=0.0,pad_post=0.0,agc=False)
    configs=json.loads(sys.argv[2]); conds=sys.argv[3].split(",")
    for name,(over,model) in configs.items():
        cfg=dict(BASE); cfg.update(over); row=[]
        for c in conds:
            hyp=norm(" ".join(run(f"{d}/{c}.wav", cfg, model)))
            S,D,I=align(ref,hyp); n=len(ref)
            row.append(f"{c}: WER {100*(S+D+I)/n:4.1f}% 漏词 {100*D/n:4.1f}%")
        print(f"{name:28s} | " + " | ".join(row), flush=True)
