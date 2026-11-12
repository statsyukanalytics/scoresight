"""Read a configured broadcast clock using ScoreSight, preserving sampled PTS."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
import numpy as np
import onnxruntime as ort
from .ppocr_engine import Recognizer, SMALL, model_files


from .readings import parse_reading
from .field_layouts import profile_layouts, merge_readings, CLOCK_CHARACTERS, PERIOD_CHARACTERS
from .runtime import batch_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--config', type=Path, help='Optional manual layout; omit for automatic detection')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, help='Limit decoding for a pilot run')
    ap.add_argument('--start-seconds', type=float, default=0,
                    help='Sample a source-video window while preserving original frame PTS')
    ap.add_argument('--ffmpeg', default='ffmpeg')
    args = ap.parse_args()
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        ap.error('--start-seconds must be finite and non-negative')
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        ap.error('--seconds must be finite and positive')
    run_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("extract.py", "auto_fields.py", "field_layouts.py", "readings.py", "runtime.py")}
    automatic = args.config is None
    cfg = json.loads(args.config.read_text()) if args.config else {'name': 'auto', 'sample_hz': 2, 'min_confidence': .9}
    args.output.mkdir(parents=True, exist_ok=True)
    if cfg['sample_hz'] <= 0:
        raise ValueError('sample_hz must be positive')
    layouts = [] if automatic else profile_layouts(cfg)
    frames_per_batch = batch_frames()
    if (args.output/'observations.csv').exists():
        raise SystemExit('Output exists; use a new output directory')
    probe = json.loads(subprocess.check_output([str(Path(args.ffmpeg).with_name('ffprobe.exe' if os.name == 'nt' else 'ffprobe')),
        '-v','error','-show_streams','-show_format','-of','json',str(args.video)], creationflags=0x08000000 if os.name=='nt' else 0))
    stream = next(s for s in probe['streams'] if s['codec_type']=='video')
    if automatic:
        cfg.update(width=stream['width'], height=stream['height'])
        from .auto_layout import detector, MODEL_REVISION, MODEL_SHA256
        from .auto_fields import AutoReader
        layout_detector = detector()
        cfg.update(layout_model_revision=MODEL_REVISION, layout_model_sha256=MODEL_SHA256)
    if (stream['width'],stream['height']) != (cfg['width'],cfg['height']):
        raise SystemExit('Video dimensions do not match configured ROIs')
    started = time.perf_counter()
    rec = Recognizer(SMALL)
    if not rec.load():
        raise RuntimeError(rec.error)
    if automatic:
        auto_reader = AutoReader(layout_detector, rec)
    # Decode sequentially; retain selected frames' original PTS instead of inventing a CFR clock.
    # Crop the shared bounding rectangle before transferring frames to Python.
    boxes = [[0, 0, cfg['width'], cfg['height']]] if automatic else [box for pair in layouts for box in pair]
    left=min(b[0] for b in boxes); top=min(b[1] for b in boxes)
    width=max(b[0]+b[2] for b in boxes)-left; height=max(b[1]+b[3] for b in boxes)-top
    cmd=[args.ffmpeg,'-hide_banner','-nostats']
    if args.start_seconds:
        cmd += ['-copyts', '-start_at_zero', '-ss', str(args.start_seconds)]
    if args.seconds: cmd += ['-t',str(args.seconds)]
    cmd += ['-i',str(args.video)]
    cmd += ['-an','-vf',f"select='gte(t,{args.start_seconds})*(isnan(prev_selected_t)+gte(t-prev_selected_t,{1/cfg['sample_hz']}))',showinfo,crop={width}:{height}:{left}:{top}:exact=1",
            '-fps_mode','passthrough','-pix_fmt','bgr24','-f','rawvideo','pipe:1']
    readings=[]; crops=[]; ocr_seconds=0; detected_layouts=[]; layout_seconds=0; frames=[]
    last_progress = started
    def flush_frames():
        nonlocal last_progress
        if frames:
            for reading, fields in auto_reader.read_many(frames):
                detected_layouts.append(fields)
                readings.append(reading)
            frames.clear()
        now = time.perf_counter()
        if now - last_progress >= 10:
            print(json.dumps({'stage': 'ocr', 'samples': len(readings),
                              'wall_seconds': round(now-started, 1)}), flush=True)
            last_progress = now
    def flush():
        nonlocal ocr_seconds
        if crops:
            t=time.perf_counter()
            result=rec.recognize(crops,[CLOCK_CHARACTERS, PERIOD_CHARACTERS] * (len(crops)//2))
            ocr_seconds += time.perf_counter()-t
            for offset in range(0, len(result), 2*len(layouts)):
                reading, fields = merge_readings(layouts, result[offset:offset+2*len(layouts)], cfg['min_confidence'])
                readings.append(reading); detected_layouts.append(fields)
            crops.clear()
    with (args.output/'decode.log').open('w+',encoding='utf8') as log:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log,creationflags=0x08000000 if os.name=='nt' else 0)
        try:
            while True:
                raw=proc.stdout.read(width*height*3)
                if not raw: break
                if len(raw)!=width*height*3: raise RuntimeError('Truncated decoded frame')
                image=np.frombuffer(raw,np.uint8).reshape(height,width,3)
                if automatic:
                    frames.append(image)
                    if len(frames) >= frames_per_batch: flush_frames()
                else:
                    for x,y,w,h in boxes: crops.append(image[y-top:y-top+h,x-left:x-left+w].copy())
                if not automatic and len(crops)>=2*len(layouts)*frames_per_batch: flush()
            if automatic: flush_frames()
            flush()
            if proc.wait()!=0: raise RuntimeError('FFmpeg failed; see decode.log')
        finally:
            proc.stdout.close()
            if proc.poll() is None: proc.kill(); proc.wait()
        log.flush();log.seek(0);decode_text=log.read()
    if automatic:
        layout_seconds = auto_reader.layout_seconds
        ocr_seconds = auto_reader.ocr_seconds
    checkpoint={'readings':readings,'ocr_seconds':ocr_seconds,'config':cfg,
                'video':str(args.video.resolve()),'source_size':args.video.stat().st_size,
                'source_mtime_ns':args.video.stat().st_mtime_ns}
    with (args.output/'raw-readings.json').open('w',encoding='utf8') as checkpoint_file:
        json.dump(checkpoint,checkpoint_file)
    pts=[float(x) for x in re.findall(r'\bpts_time:([\d.eE+-]+)',decode_text)]
    if len(pts)!=len(readings): raise RuntimeError(f'PTS/readings mismatch: {len(pts)} vs {len(readings)}')
    (args.output/'layout-observations.json').write_text(json.dumps(
        [{'video_seconds': t, 'fields': fields} for t, fields in zip(pts, detected_layouts)]), encoding='utf8')
    valid_count=0; previous=None; segment=0
    with (args.output/'observations.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f)
        writer.writerow(['video_seconds','clock_raw','period_raw','clock_confidence','period_confidence','clock_seconds','period','valid','sequence_id','transition'])
        for timestamp,(clock,period) in zip(pts,readings):
            confidence=min(clock[1] or 0,period[1] or 0)
            seconds,p,valid=parse_reading(clock[0],period[0],confidence,cfg['min_confidence'])
            transition='unreadable'
            if valid:
                valid_count+=1; transition='start'
                if previous:
                    t0,s0,p0=previous; delta=s0-seconds
                    transition='continuous' if p==p0 and 0<=delta<=timestamp-t0+2 else 'discontinuity'
                    if timestamp-t0>2: transition='gap'
                    if transition!='continuous': segment+=1
                previous=(timestamp,seconds,p)
            writer.writerow([timestamp,clock[0],period[0],clock[1],period[1],seconds,p,valid,segment,transition])
    summary={'video':str(args.video.resolve()),'source_size':args.video.stat().st_size,'source_mtime_ns':args.video.stat().st_mtime_ns,
             'duration_seconds':float(probe['format']['duration']),
             'config':cfg,'start_seconds':args.start_seconds,'requested_seconds':args.seconds,'script_sha256':run_hashes['extract.py'],'code_hashes':run_hashes,
             'parser_sha256':run_hashes['readings.py'],
             'runtime':{'onnxruntime':ort.__version__,'numpy':np.__version__},
             'observations':len(readings),'valid':valid_count,'ocr_seconds':ocr_seconds,'wall_seconds':time.perf_counter()-started,
             'layout_seconds':layout_seconds,'batch_frames':frames_per_batch,'layout_policy':'validated alternate layouts; reject conflicting readings','layout_detected':sum(fields is not None for fields in detected_layouts),
             'timestamp_basis':'selected decoded frame PTS; FFmpeg default input start normalization','provider':rec._session.get_providers(),
             'status':'unreviewed OCR; sequence flags do not classify live play or replay'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
