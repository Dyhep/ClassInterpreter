"""Synthesize a lecture (teacher voice) + classroom babble, and build 3 listening conditions."""
import numpy as np, soundfile as sf, sherpa_onnx, random, json, sys
random.seed(7); np.random.seed(7)
T="/home/claude/models/tts/"
def tts(name):
    d=T+name+"/"; onnx=[f for f in __import__("os").listdir(d) if f.endswith(".onnx")][0]
    cfg=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
        model=d+onnx, tokens=d+"tokens.txt", data_dir=d+"espeak-ng-data"), num_threads=2))
    return sherpa_onnx.OfflineTts(cfg)
def say(t, text, speed=1.0):
    a=t.generate(text, sid=0, speed=speed); x=np.array(a.samples, dtype=np.float32)
    if a.sample_rate!=16000:
        n=int(len(x)*16000/a.sample_rate); x=np.interp(np.linspace(0,len(x)-1,n), np.arange(len(x)), x).astype(np.float32)
    return x
lines=[l.strip() for l in open(sys.argv[1]) if l.strip()]
teacher=tts("vits-piper-en_US-lessac-medium")
parts=[]; ref=[]
for l in lines:
    parts.append(say(teacher, l, speed=1.12)); ref.append(l)
    cont = l.endswith(",") or not l[-1] in ".?!"
    gap = random.uniform(0.12,0.25) if cont else random.choice([random.uniform(0.35,0.8), random.uniform(0.8,1.6)])
    parts.append(np.zeros(int(gap*16000), np.float32))
clean=np.concatenate(parts); clean/=np.abs(clean).max()+1e-9; clean*=0.7
# classroom babble: two other voices saying other things, chopped and overlapped
others=[tts("vits-piper-en_US-amy-low"), tts("vits-piper-en_GB-alan-low")]
chat=["Did you finish the reading for today?","I think the quiz is next week, right?","Can I borrow a pen?","What page are we on?",
      "I didn't understand the last part either.","Are you going to the library after this?","My laptop battery is almost dead.",
      "Wait, what did he just say?","Let's study together on Friday.","Is this going to be on the exam?"]
bab=np.zeros_like(clean)
for o in others:
    pos=int(np.random.uniform(0,3)*16000)
    while pos < len(bab):
        x=say(o, random.choice(chat), speed=random.uniform(0.95,1.2))*random.uniform(0.3,0.8)
        e=min(len(bab), pos+len(x)); bab[pos:e]+=x[:e-pos]; pos=e+int(random.uniform(0.5,4)*16000)
def reverb(x, rt60):
    n=int(rt60*16000); t=np.arange(n)/16000; ir=np.random.randn(n)*np.exp(-6.9*t/rt60); ir[0]=1; ir/=np.sqrt((ir**2).sum())
    y=np.convolve(x, ir)[:len(x)]; return (y/ (np.abs(y).max()+1e-9)).astype(np.float32)
def bandpass(x, lo, hi):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/16000); X[(f<lo)|(f>hi)]*=0.05; return np.fft.irfft(X,len(x)).astype(np.float32)
def pink(n):
    w=np.fft.rfft(np.random.randn(n)); f=np.arange(len(w)); f[0]=1; return np.fft.irfft(w/np.sqrt(f), n).astype(np.float32)
def at_snr(sig, noise, snr):
    ps=np.mean(sig**2); pn=np.mean(noise**2)+1e-12; return noise*np.sqrt(ps/(pn*10**(snr/10)))
def rms(x): return np.sqrt(np.mean(x**2))
out={}
out["quiet"]=clean
room=bandpass(reverb(clean,0.35),180,7500)            # iPad speaker -> laptop mic in a small room
room=room+at_snr(room, pink(len(room)), 30); out["room"]=room*0.5
far=reverb(clean,0.9)                                  # teacher far away in a lecture hall
far=far+at_snr(far, reverb(bab,0.6), 8)+at_snr(far, pink(len(far)), 18)
out["classroom"]=far/np.abs(far).max()*0.12             # and quiet: far-field level
for k,v in out.items(): sf.write(f"{sys.argv[2]}/{k}.wav", v.astype(np.float32), 16000); print(k, f"{len(v)/16000:.0f}s rms={rms(v):.4f}")
json.dump(ref, open(f"{sys.argv[2]}/ref.json","w"))
