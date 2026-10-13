import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import json,sys,time
from pathlib import Path
from mlx_vlm import load,generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
root=Path("/Volumes/ssd/minicpm-v47-eval")
repo=Path("/Users/anshi/clawd/lingbot-world-v2")
model_dir=Path("/Volumes/ssd/huggingface/hub/models--mlx-community--MiniCPM-V-4.6-4bit/snapshots/86cd463d33a946e4481b77e3c10fc63121b60a19")
scene_id=sys.argv[1]
scene=next(x for x in json.loads((repo/"eval/g0.7/config.json").read_text())["scenes"] if x["id"]==scene_id)
prompt=(root/"world_perception_prompt_v2b.txt").read_text()+"\nTarget JSON schema:\n"+(root/"world_schema.json").read_text()
r={"scene":scene_id,"model":"MiniCPM-V 4.6 MLX 4bit","same_decoder":False,"max_new_tokens":550,"temperature":0,"seed":42}
start=time.monotonic()
try:
 t=time.monotonic()
 model,processor=load(str(model_dir))
 config=load_config(str(model_dir))
 r["load_s"]=round(time.monotonic()-t,3)
 messages=[{"role":"user","content":[{"type":"image","image":str(repo/scene["image"])},{"type":"text","text":prompt}]}]
 formatted=apply_chat_template(processor,config,messages,tokenize=False,enable_thinking=False)
 t=time.monotonic()
 out=generate(model,processor,formatted,image=str(repo/scene["image"]),max_tokens=550,temperature=0,seed=42,enable_thinking=False,thinking_budget=0,verbose=False)
 r["inference_s"]=round(time.monotonic()-t,3)
 r["raw_text"]=out.text if hasattr(out,"text") else str(out)
 r["generation_tokens"]=getattr(out,"generation_tokens",None)
 r["generation_tps"]=getattr(out,"generation_tps",None)
 raw=r["raw_text"]; pos=raw.find("{")
 try:
  r["world"]=json.JSONDecoder().raw_decode(raw[pos:])[0] if pos>=0 else None
 except Exception as exc:r["json_error"]=repr(exc)
except Exception as exc:r["error"]=repr(exc)
r["total_s"]=round(time.monotonic()-start,3)
dest=root/("g07_46_mlx_v2b_"+scene_id+".json")
with dest.open("x") as f:json.dump(r,f,indent=2,ensure_ascii=False)
print(json.dumps({k:v for k,v in r.items() if k not in ("raw_text","world")},ensure_ascii=False))
print("RAW:",r.get("raw_text","")[:1500])
