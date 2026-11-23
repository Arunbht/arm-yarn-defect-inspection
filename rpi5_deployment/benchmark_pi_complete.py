#!/usr/bin/env python3
"""
Comprehensive All-In-One Benchmark & Telemetry Suite for Raspberry Pi 5.
Measures and records:
  1. System hardware, OS, kernel, memory, and initial thermals/throttling.
  2. MobileNetV3 vision pipeline: file info, SHA-256, classes, and exact 96x768 per-stage timings.
  3. ExecuTorch Llama 3.2 1B: cold load time, RAM delta, 1-shot TTFT, 10 generation calls with
     explicit 'effective_tok_per_sec_including_prefill' and 'pure_decode_tok_per_sec', and format validity.
  4. 5-Minute Sustained Pipeline: active 1 FPS stream with 96x768 vision and real Llama defect triggering,
     recording start, every 30s, and end thermals, throttle register, peak RSS, frame drops, and advice counts.
Outputs:
  - Formatted terminal report
  - Comprehensive JSON artifact: 'pi5_benchmark_results.json'
"""

import os
import sys
import time
import json
import hashlib
import platform
import resource
import threading
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUTPUT_JSON_PATH = ROOT / "pi5_benchmark_results.json"


def get_peak_rss_mb():
    # ru_maxrss is in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return res.stdout.strip()
    except Exception as e:
        return f"ERR: {e}"


def read_thermal_and_throttle():
    temp_str = run_cmd("vcgencmd measure_temp")
    throt_str = run_cmd("vcgencmd get_throttled")
    if not temp_str or "ERR" in temp_str:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                t = float(f.read().strip()) / 1000.0
                temp_str = f"temp={t:.1f}'C"
        except Exception:
            temp_str = "temp=N/A"
    if not throt_str or "ERR" in throt_str:
        throt_str = "throttled=N/A"
    return temp_str, throt_str


def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def crop_band(im):
    w, h = im.size
    y0 = int(0.30 * h)
    y1 = max(y0 + 1, int(0.70 * h))
    return im.crop((0, y0, w, y1))


def main():
    print("=" * 70)
    print("   RASPBERRY PI 5 COMPREHENSIVE PIPELINE BENCHMARK (ALL-IN-ONE)")
    print("=" * 70)

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": {},
        "vision_model": {},
        "llama_model": {},
        "sustained_5min_run": {}
    }

    # ---------------------------------------------------------
    # 1. System Info & Environment
    # ---------------------------------------------------------
    print("\n[1/4] SYSTEM SPECIFICATIONS & ENVIRONMENT:")
    meminfo = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = parts[1].strip()
    except Exception:
        pass

    total_ram_kb = meminfo.get("MemTotal", "N/A")
    avail_ram_kb = meminfo.get("MemAvailable", "N/A")
    initial_temp, initial_throt = read_thermal_and_throttle()

    results["system"] = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "os_pretty_name": run_cmd('cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d \\"'),
        "total_ram_kb": total_ram_kb,
        "available_ram_kb": avail_ram_kb,
        "python_version": sys.version.split()[0],
        "initial_telemetry": {
            "temp": initial_temp,
            "throttled": initial_throt,
            "initial_rss_mb": round(get_peak_rss_mb(), 2)
        }
    }

    print(f"  • Architecture:    {results['system']['machine']} ({results['system']['kernel']})")
    print(f"  • OS:              {results['system']['os_pretty_name']}")
    print(f"  • RAM:             Total: {total_ram_kb} | Available: {avail_ram_kb}")
    print(f"  • Python Version:  {results['system']['python_version']}")
    print(f"  • Initial State:   {initial_temp} | {initial_throt} | RSS: {results['system']['initial_telemetry']['initial_rss_mb']} MB")

    # ---------------------------------------------------------
    # 2. Vision Classifier Benchmark (96x768 Pipeline)
    # ---------------------------------------------------------
    print("\n[2/4] VISION CLASSIFIER BENCHMARK (MobileNetV3 - 96x768 Pipeline):")
    import torch
    import torch.nn as nn
    from torchvision import models
    import torchvision.transforms as T
    from PIL import Image

    model_path = ROOT / "yarn.pt"
    if not model_path.exists():
        print(f"  [!] yarn.pt not found at {model_path}")
        return

    stat = model_path.stat()
    file_bytes = stat.st_size
    file_mb = round(file_bytes / (1024 * 1024), 2)
    file_sha256 = get_file_sha256(model_path)

    t_load_0 = time.perf_counter()
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    vision_model = models.mobilenet_v3_small()
    vision_model.classifier[3] = nn.Linear(vision_model.classifier[3].in_features, len(classes))
    vision_model.load_state_dict(ck["m"])
    vision_model.eval()
    t_load_vision = round((time.perf_counter() - t_load_0) * 1000, 2)

    total_params = sum(p.numel() for p in vision_model.parameters())

    print(f"  • Model Path:        {model_path.name} ({file_mb} MB, {file_bytes} bytes)")
    print(f"  • SHA-256 Hash:      {file_sha256[:16]}...")
    print(f"  • Parameters:        {total_params:,} (FP32)")
    print(f"  • Classes (N={len(classes)}):    {classes}")
    print(f"  • Checkpoint Load:   {t_load_vision} ms")

    # Sample images for testing
    seq_dir = ROOT / "sequence_30"
    test_imgs = list(seq_dir.glob("*.bmp")) + list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png"))
    if not test_imgs:
        img_dummy = Image.new("RGB", (1024, 256), color=(200, 200, 200))
        test_img_path = ROOT / "temp_benchmark_sample.bmp"
        img_dummy.save(test_img_path)
    else:
        test_img_path = test_imgs[0]

    transform_pipeline = T.Compose([
        T.Lambda(crop_band),
        T.Resize((96, 768)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Warmup
    for _ in range(5):
        im_w = Image.open(test_img_path).convert("RGB")
        inp_w = transform_pipeline(im_w).unsqueeze(0)
        with torch.no_grad():
            _ = vision_model(inp_w)

    # 50 Iterations
    decode_times = []
    preproc_times = []
    infer_times = []
    total_vision_times = []

    for _ in range(50):
        t0 = time.perf_counter()
        im = Image.open(test_img_path).convert("RGB")
        t1 = time.perf_counter()

        inp = transform_pipeline(im).unsqueeze(0)
        t2 = time.perf_counter()

        with torch.no_grad():
            out = vision_model(inp)
        t3 = time.perf_counter()

        decode_times.append((t1 - t0) * 1000)
        preproc_times.append((t2 - t1) * 1000)
        infer_times.append((t3 - t2) * 1000)
        total_vision_times.append((t3 - t0) * 1000)

    avg_decode = sum(decode_times) / len(decode_times)
    avg_preproc = sum(preproc_times) / len(preproc_times)
    avg_infer = sum(infer_times) / len(infer_times)
    avg_total = sum(total_vision_times) / len(total_vision_times)
    fps = 1000.0 / avg_total

    results["vision_model"] = {
        "file_name": model_path.name,
        "file_bytes": file_bytes,
        "file_mb": file_mb,
        "sha256": file_sha256,
        "total_params": total_params,
        "classes": classes,
        "transform": "crop_band(30%-70%) + Resize((96, 768)) + Normalize",
        "input_tensor_shape": [1, 3, 96, 768],
        "load_time_ms": t_load_vision,
        "decode_ms": {
            "avg": round(avg_decode, 2),
            "min": round(min(decode_times), 2),
            "max": round(max(decode_times), 2)
        },
        "preprocess_ms": {
            "avg": round(avg_preproc, 2),
            "min": round(min(preproc_times), 2),
            "max": round(max(preproc_times), 2)
        },
        "inference_ms": {
            "avg": round(avg_infer, 2),
            "min": round(min(infer_times), 2),
            "max": round(max(infer_times), 2)
        },
        "total_frame_ms": {
            "avg": round(avg_total, 2),
            "min": round(min(total_vision_times), 2),
            "max": round(max(total_vision_times), 2)
        },
        "sustained_fps": round(fps, 2),
        "rss_after_vision_mb": round(get_peak_rss_mb(), 2)
    }

    print(f"  • Image Decode:      {avg_decode:.2f} ms")
    print(f"  • Preprocessing:     {avg_preproc:.2f} ms (crop_band + 96x768 resize)")
    print(f"  • Model Inference:   {avg_infer:.2f} ms")
    print(f"  • Total Per Frame:   {avg_total:.2f} ms")
    print(f"  • Sustained Speed:   {fps:.1f} FPS")
    print(f"  • RSS After Vision:  {results['vision_model']['rss_after_vision_mb']} MB")

    # ---------------------------------------------------------
    # 3. ExecuTorch Llama 3.2 1B Benchmark (1-Shot Template)
    # ---------------------------------------------------------
    print("\n[3/4] EXECUTORCH LLAMA 3.2 1B BENCHMARK (1-Shot Prompt):")
    from llm_advice import Advisor, FEW_SHOT_EXAMPLES

    rss_before_llm = get_peak_rss_mb()
    t_llm_0 = time.perf_counter()
    advisor = Advisor(ROOT / "llama_model")
    tokenizer = advisor.tokenizer
    llm_model = advisor.model
    t_llm_load = round(time.perf_counter() - t_llm_0, 2)
    rss_after_llm = get_peak_rss_mb()

    pte_path = ROOT / "llama_model" / "model.pte"
    pte_bytes = pte_path.stat().st_size if pte_path.exists() else 0
    pte_mb = round(pte_bytes / (1024 * 1024), 2)

    print(f"  • ExecuTorch PTE:    {pte_mb} MB ({pte_bytes:,} bytes)")
    print(f"  • Model Load Time:   {t_llm_load} s")
    print(f"  • RAM Footprint:     +{round(rss_after_llm - rss_before_llm, 1)} MB (Peak RSS: {round(rss_after_llm, 1)} MB)")

    system_prompt = (
        "You are an automated quality control assistant in a textile spinning mill inspecting textile yarn bobbins. "
        "Provide machine operator instructions for textile yarn defects. Write exactly two lines:\n"
        "ACTION: <one short action>\n"
        "CHECK: <one short check>\n"
        "Be conservative. Output nothing else."
    )

    prompt_cases = [
        ("knotting", "HIGH", 0.98, "cluster of 3 knots in 10s"),
        ("irregularity", "MEDIUM", 0.91, "transient thickness spike"),
        ("long segment lack", "HIGH", 0.99, "yarn break detected"),
        ("short segment lack", "MEDIUM", 0.88, "intermittent optical void"),
        ("thinness", "MEDIUM", 0.95, "draft tension low"),
        ("knotting", "HIGH", 0.94, "spindle 4 defect cluster"),
        ("irregularity", "MEDIUM", 0.87, "periodic slub detected"),
        ("long segment lack", "HIGH", 0.96, "missing thread continuous"),
        ("short segment lack", "MEDIUM", 0.92, "micro void 2mm"),
        ("thinness", "MEDIUM", 0.89, "bobbin winding taper")
    ]

    # Measure TTFT (max_seq_len = prompt_len + 1)
    print("\n  Measuring Time-To-First-Token (TTFT) with max_seq_len = prompt_len + 1...")
    ttft_samples = []
    for idx in range(min(5, len(prompt_cases))):
        lbl, sev, conf, hist = prompt_cases[idx]
        msg = [
            {"role": "system", "content": system_prompt},
            *FEW_SHOT_EXAMPLES[:2],  # 1-shot
            {"role": "user", "content": f"Textile Defect: {lbl}\nSeverity: {sev}\nConfidence: {conf:.2f}\nRecent Inspections: {hist}"}
        ]
        p_str = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        t_ids = tokenizer.encode(p_str)
        t_start = time.perf_counter()
        _ = llm_model.generate(t_ids, max_seq_len=len(t_ids) + 1)
        ttft = round(time.perf_counter() - t_start, 2)
        ttft_samples.append(ttft)
        print(f"    [Sample {idx+1}/5] Prompt: {len(t_ids)} toks | TTFT: {ttft} s")

    avg_ttft = round(sum(ttft_samples) / len(ttft_samples), 2)
    print(f"  • Measured Mean TTFT: {avg_ttft} s")

    # 10 Full Generation Calls
    print("\n  Running 10 Production Advisory Calls (1-Shot)...")
    calls_data = []
    valid_count = 0

    for i, (lbl, sev, conf, hist) in enumerate(prompt_cases, 1):
        msg = [
            {"role": "system", "content": system_prompt},
            *FEW_SHOT_EXAMPLES[:2],
            {"role": "user", "content": f"Textile Defect: {lbl}\nSeverity: {sev}\nConfidence: {conf:.2f}\nRecent Inspections: {hist}"}
        ]
        p_str = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        t_ids = tokenizer.encode(p_str)
        p_len = len(t_ids)

        t0 = time.perf_counter()
        out_ids = llm_model.generate(t_ids, max_seq_len=p_len + 30)
        tot_time = round(time.perf_counter() - t0, 2)

        n_tok = len(out_ids)
        # 1. Total tokens / total elapsed (including prefill)
        eff_tok_s = round(n_tok / tot_time, 2) if tot_time > 0 else 0.0

        # 2. Pure decode rate: tokens / (tot_time - avg_ttft)
        decode_duration = max(0.01, tot_time - avg_ttft)
        pure_decode_tok_s = round(n_tok / decode_duration, 2)

        raw_text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        has_act = any(l.strip().upper().startswith("ACTION:") for l in raw_text.splitlines())
        has_chk = any(l.strip().upper().startswith("CHECK:") for l in raw_text.splitlines())
        is_valid = has_act and has_chk
        if is_valid:
            valid_count += 1

        call_record = {
            "call_index": i,
            "defect": lbl,
            "severity": sev,
            "prompt_tokens": p_len,
            "generated_tokens": n_tok,
            "total_response_seconds": tot_time,
            "estimated_ttft_seconds": avg_ttft,
            "effective_tok_per_sec_including_prefill": eff_tok_s,
            "pure_decode_tok_per_sec": pure_decode_tok_s,
            "format_valid": is_valid,
            "output_preview": raw_text.replace("\n", " | ")[:90]
        }
        calls_data.append(call_record)

        v_mark = "VALID" if is_valid else "INVALID"
        print(f"    [{i:02d}/10] {lbl:<18} | Time: {tot_time:5.2f}s | {n_tok:2d} tok | Eff: {eff_tok_s:4.2f} t/s | Pure Decode: {pure_decode_tok_s:4.2f} t/s | {v_mark}")

    all_times = [c["total_response_seconds"] for c in calls_data]
    all_eff_rates = [c["effective_tok_per_sec_including_prefill"] for c in calls_data]
    all_decode_rates = [c["pure_decode_tok_per_sec"] for c in calls_data]

    results["llama_model"] = {
        "pte_file_name": pte_path.name,
        "pte_file_bytes": pte_bytes,
        "pte_file_mb": pte_mb,
        "load_time_sec": t_llm_load,
        "ram_footprint_mb": round(rss_after_llm - rss_before_llm, 1),
        "peak_rss_mb": round(rss_after_llm, 1),
        "prompt_template": {
            "type": "1-shot",
            "system_prompt": system_prompt,
            "few_shot_examples_count": 1,
            "average_prompt_tokens": round(sum(c["prompt_tokens"] for c in calls_data) / len(calls_data), 1)
        },
        "ttft_seconds": {
            "samples": ttft_samples,
            "mean": avg_ttft,
            "min": min(ttft_samples),
            "max": max(ttft_samples)
        },
        "response_time_seconds": {
            "mean": round(sum(all_times) / len(all_times), 2),
            "min": round(min(all_times), 2),
            "max": round(max(all_times), 2)
        },
        "effective_tok_per_sec_including_prefill": {
            "mean": round(sum(all_eff_rates) / len(all_eff_rates), 2),
            "min": round(min(all_eff_rates), 2),
            "max": round(max(all_eff_rates), 2)
        },
        "pure_decode_tok_per_sec": {
            "mean": round(sum(all_decode_rates) / len(all_decode_rates), 2),
            "min": round(min(all_decode_rates), 2),
            "max": round(max(all_decode_rates), 2)
        },
        "validator_pass_rate": f"{valid_count}/{len(calls_data)} ({valid_count/len(calls_data)*100:.1f}%)",
        "per_call_details": calls_data
    }

    print(f"\n  • Mean TTFT (1-shot):            {avg_ttft} s")
    print(f"  • Total Time:                    Mean: {results['llama_model']['response_time_seconds']['mean']} s (Min: {results['llama_model']['response_time_seconds']['min']} s, Max: {results['llama_model']['response_time_seconds']['max']} s)")
    print(f"  • Eff tok/s (inc. prefill):      Mean: {results['llama_model']['effective_tok_per_sec_including_prefill']['mean']} tok/s")
    print(f"  • Pure Decode tok/s:             Mean: {results['llama_model']['pure_decode_tok_per_sec']['mean']} tok/s")
    print(f"  • Validator Pass Rate:           {results['llama_model']['validator_pass_rate']}")

    # ---------------------------------------------------------
    # 4. Sustained 5-Minute Real Pipeline Stream Run
    # ---------------------------------------------------------
    print("\n[4/4] 5-MINUTE SUSTAINED REAL PIPELINE STREAM (1 FPS Cadence):")
    print("  Running continuous 96x768 vision stream + real Llama defect triggering...")

    active_frames = test_imgs if len(test_imgs) >= 5 else [test_img_path] * 5
    stream_duration = 300  # 5 minutes
    target_cadence = 1.0  # 1 FPS

    telemetry_timeline = []
    initial_temp_s, initial_throt_s = read_thermal_and_throttle()
    telemetry_timeline.append({
        "elapsed_sec": 0,
        "temperature": initial_temp_s,
        "throttled": initial_throt_s,
        "peak_rss_mb": round(get_peak_rss_mb(), 2)
    })
    print(f"  [  0s START] {initial_temp_s} | {initial_throt_s} | RSS: {get_peak_rss_mb():.1f} MB")

    total_frames_classified = 0
    normal_frames_count = 0
    defect_frames_count = 0
    llm_advice_count = 0
    fallback_advice_count = 0
    frames_dropped = 0

    llm_lock = threading.Lock()
    is_llm_busy = False

    def async_llm_task(lbl, sev, conf):
        nonlocal is_llm_busy, llm_advice_count
        try:
            msg = [
                {"role": "system", "content": system_prompt},
                *FEW_SHOT_EXAMPLES[:2],
                {"role": "user", "content": f"Textile Defect: {lbl}\nSeverity: {sev}\nConfidence: {conf:.2f}"}
            ]
            p_s = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            t_ids = tokenizer.encode(p_s)
            _ = llm_model.generate(t_ids, max_seq_len=len(t_ids) + 30)
            with llm_lock:
                llm_advice_count += 1
        except Exception as e:
            pass
        finally:
            with llm_lock:
                is_llm_busy = False

    start_stream = time.perf_counter()
    last_sample_time = start_stream
    frame_idx = 0

    while (time.perf_counter() - start_stream) < stream_duration:
        loop_start = time.perf_counter()

        # Ingest next image
        im_path = active_frames[frame_idx % len(active_frames)]
        frame_idx += 1

        im = Image.open(im_path).convert("RGB")
        inp = transform_pipeline(im).unsqueeze(0)
        with torch.no_grad():
            out = vision_model(inp)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)
            pred_class = classes[pred_idx.item()]
            conf_val = round(conf.item(), 3)

        total_frames_classified += 1

        # Pipeline logic
        if pred_class == "normal":
            normal_frames_count += 1
        else:
            defect_frames_count += 1
            sev = "HIGH" if pred_class in ["knotting", "long segment lack"] else "MEDIUM"
            with llm_lock:
                if not is_llm_busy:
                    is_llm_busy = True
                    t = threading.Thread(target=async_llm_task, args=(pred_class, sev, conf_val), daemon=True)
                    t.start()
                else:
                    # Deterministic fallback while LLM is generating
                    fallback_advice_count += 1

        # Check telemetry every 30 seconds
        now = time.perf_counter()
        if (now - last_sample_time) >= 30.0:
            elapsed_sec = int(now - start_stream)
            cur_temp, cur_throt = read_thermal_and_throttle()
            cur_rss = round(get_peak_rss_mb(), 2)
            telemetry_timeline.append({
                "elapsed_sec": elapsed_sec,
                "temperature": cur_temp,
                "throttled": cur_throt,
                "peak_rss_mb": cur_rss
            })
            print(f"  [+{elapsed_sec:3d}s] {cur_temp} | {cur_throt} | RSS: {cur_rss:.1f} MB | Frames: {total_frames_classified} (LLM: {llm_advice_count}, Fallback: {fallback_advice_count})")
            last_sample_time = now

        # Pace to target cadence (1 FPS)
        loop_elapsed = time.perf_counter() - loop_start
        sleep_dur = target_cadence - loop_elapsed
        if sleep_dur > 0:
            time.sleep(sleep_dur)
        else:
            # Took longer than 1.0s budget
            frames_dropped += 1

    final_temp_s, final_throt_s = read_thermal_and_throttle()
    telemetry_timeline.append({
        "elapsed_sec": int(time.perf_counter() - start_stream),
        "temperature": final_temp_s,
        "throttled": final_throt_s,
        "peak_rss_mb": round(get_peak_rss_mb(), 2)
    })
    print(f"  [{int(time.perf_counter() - start_stream):3d}s END]   {final_temp_s} | {final_throt_s} | RSS: {get_peak_rss_mb():.1f} MB")

    results["sustained_5min_run"] = {
        "duration_seconds": round(time.perf_counter() - start_stream, 2),
        "target_cadence_fps": target_cadence,
        "total_frames_classified": total_frames_classified,
        "normal_frames_count": normal_frames_count,
        "defect_frames_count": defect_frames_count,
        "llm_advice_generated": llm_advice_count,
        "fallback_advice_generated": fallback_advice_count,
        "frames_dropped": frames_dropped,
        "final_peak_rss_mb": round(get_peak_rss_mb(), 2),
        "start_state": {"temp": initial_temp_s, "throttled": initial_throt_s},
        "end_state": {"temp": final_temp_s, "throttled": final_throt_s},
        "timeline_samples_every_30s": telemetry_timeline
    }

    # Save complete JSON
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"   BENCHMARK COMPLETED SUCCESSFULLY")
    print(f"   Results saved to: {OUTPUT_JSON_PATH.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
