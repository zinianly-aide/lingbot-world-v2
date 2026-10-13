import json,re,statistics,copy
from pathlib import Path
R=Path("/Volumes/ssd/minicpm-v47-eval")
C=json.loads(Path("/Users/anshi/clawd/lingbot-world-v2/eval/g0.7/config.json").read_text())
keys={"environment","scene_layout","main_entities","lighting","weather","camera","motion","persistent_constraints","user_intent"}
rows=[]
for scene in C["scenes"]:
 id=scene["id"]
 raw=json.loads((R/"logs"/f"g07_v2b_{id}.stdout.log").read_text())
 err=(R/"logs"/f"g07_v2b_{id}.stderr.log").read_text()
 v46=json.loads((R/f"g07_46_mlx_v2b_{id}.json").read_text())
 t=re.search(r"([\d.]+) real",err)
 vision=re.search(r"mtmd batch encoding done in (\d+) ms",err)
 rss=re.search(r"(\d+)\s+maximum resident set size",err)
 fp=re.search(r"(\d+)\s+peak memory footprint",err)
 normalized=copy.deepcopy(raw)
 normalized["user_intent"]=""
 normalized["motion"]="unknown"
 normalized["camera"]=re.sub(r"\b(static|fixed)\b[,; ]*","",str(normalized["camera"]),flags=re.I).strip(" ,;") or "unknown"
 normalized["persistent_constraints"]=[x for x in normalized["persistent_constraints"] if not re.search(r"no actions|no predictions|identity and layout only|no additional objects|future|camera movement",x,re.I)]
 for e in normalized["main_entities"]:
  if re.search(r"forward|toward|moving|gliding|panning",e.get("state",""),re.I):e["state"]="unknown"
 row={"scene":id,"4_7_json":True,"4_7_keys":keys.issubset(raw),"4_7_environment":raw["environment"],
 "4_7_layout":raw["scene_layout"],"4_7_entities":[x["name"] for x in raw["main_entities"]],
 "4_7_intent_empty":raw["user_intent"]=="","4_7_motion_exact_unknown":raw["motion"]=="unknown",
 "4_7_camera_raw":raw["camera"],"4_7_visible_constraints":raw["persistent_constraints"],
 "4_7_elapsed_s":float(t.group(1)) if t else None,"4_7_vision_ms":int(vision.group(1)) if vision else None,
 "4_7_max_rss_bytes":int(rss.group(1)) if rss else None,"4_7_peak_footprint_bytes":int(fp.group(1)) if fp else None,
 "4_6_json":bool(v46.get("world")),"4_6_load_s":v46.get("load_s"),"4_6_inference_s":v46.get("inference_s"),
 "4_6_tokens":v46.get("generation_tokens"),"4_6_tokens_per_sec":v46.get("generation_tps"),
 "visual_hallucination_review":"NOT VERIFIED","spatial_accuracy_review":"NOT VERIFIED",
 "raw_world":raw,"derived_world_semantic_gate":normalized,"authoritative_user_request_separate":scene["user_prompt"]}
 rows.append(row)
summary={"n":len(rows),"v47_json":sum(x["4_7_json"] for x in rows),"v47_schema":sum(x["4_7_keys"] for x in rows),
 "v47_nonempty_entities":sum(bool(x["4_7_entities"]) for x in rows),
 "v47_empty_intent":sum(x["4_7_intent_empty"] for x in rows),
 "v47_motion_exact_unknown":sum(x["4_7_motion_exact_unknown"] for x in rows),
 "v46_json":sum(x["4_6_json"] for x in rows),
 "v47_wall_avg_s":round(statistics.mean(x["4_7_elapsed_s"] for x in rows),2),
 "v47_vision_encode_avg_ms":round(statistics.mean(x["4_7_vision_ms"] for x in rows)),
 "v47_max_rss_bytes":max(x["4_7_max_rss_bytes"] for x in rows),
 "v47_max_footprint_bytes":max(x["4_7_peak_footprint_bytes"] for x in rows),
 "v46_avg_inference_s":round(statistics.mean(x["4_6_inference_s"] for x in rows),2),
 "v46_avg_decode_tps":round(statistics.mean(x["4_6_tokens_per_sec"] for x in rows),1)}
result={"summary":summary,"comparison_limitations":["MLX 4.6 has no grammar-constrained JSON decoder here","4.6 includes schema text in prompt, 4.7 grammar schema","Tokenizer, sampling implementations and prompt preprocessing differ","True cold-start, first-token, GPU-unified-memory peaks and repeated stability NOT RUN"],"rows":rows}
with (R/"g07_v2b_ab_results.json").open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,indent=2))
for x in rows:print(x["scene"],"47 intent",x["4_7_intent_empty"],"motion",x["4_7_motion_exact_unknown"],"46_json",x["4_6_json"],"entities",x["4_7_entities"])
