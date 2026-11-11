"""Read a configured broadcast clock using ScoreSight, preserving sampled PTS."""
import argparse
import csv
import hashlib
import json
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, help='Limit decoding for a pilot run')
    ap.add_argument('--ffmpeg', default='ffmpeg')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if cfg['sample_hz'] <= 0:
        raise ValueError('sample_hz must be positive')
    for x, y, w, h in (cfg['clock'], cfg['period']):
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > cfg['width'] or y+h > cfg['height']:
            raise ValueError('ROI falls outside configured video dimensions')
    if (args.output/'observations.csv').exists():
        raise SystemExit('Output exists; use a new output directory')
    probe = json.loads(subprocess.check_output([str(Path(args.ffmpeg).with_name('ffprobe.exe' if os.name == 'nt' else 'ffprobe')),
        '-v','error','-show_streams','-show_format','-of','json',str(args.video)], creationflags=0x08000000 if os.name=='nt' else 0))
    stream = next(s for s in probe['streams'] if s['codec_type']=='video')
    if (stream['width'],stream['height']) != (cfg['width'],cfg['height']):
        raise SystemExit('Video dimensions do not match configured ROIs')
    started = time.perf_counter()
    rec = Recognizer(SMALL)
    if not rec.load():
        raise RuntimeError(rec.error)
    # Decode sequentially; retain selected frames' original PTS instead of inventing a CFR clock.
    # Crop the shared bounding rectangle before transferring frames to Python.
    boxes = [cfg['clock'],cfg['period']]
    left=min(b[0] for b in boxes); top=min(b[1] for b in boxes)
    width=max(b[0]+b[2] for b in boxes)-left; height=max(b[1]+b[3] for b in boxes)-top
    cmd=[args.ffmpeg,'-hide_banner','-nostats']
    if args.seconds: cmd += ['-t',str(args.seconds)]
    cmd += ['-i',str(args.video)]
    cmd += ['-an','-vf',f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{1/cfg['sample_hz']})',showinfo,crop={width}:{height}:{left}:{top}:exact=1",
            '-fps_mode','passthrough','-pix_fmt','bgr24','-f','rawvideo','pipe:1']
    readings=[]; crops=[]; ocr_seconds=0
    def flush():
        nonlocal ocr_seconds
        if crops:
            t=time.perf_counter()
            result=rec.recognize(crops,['0123456789:.' if i%2==0 else '0123456789STNDROstndro' for i in range(len(crops))])
            ocr_seconds += time.perf_counter()-t
            readings.extend(zip(result[::2],result[1::2])); crops.clear()
    with (args.output/'decode.log').open('w+',encoding='utf8') as log:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log,creationflags=0x08000000 if os.name=='nt' else 0)
        try:
            while True:
                raw=proc.stdout.read(width*height*3)
                if not raw: break
                if len(raw)!=width*height*3: raise RuntimeError('Truncated decoded frame')
                image=np.frombuffer(raw,np.uint8).reshape(height,width,3)
                for x,y,w,h in boxes: crops.append(image[y-top:y-top+h,x-left:x-left+w].copy())
                if len(crops)>=32: flush()
            flush()
            if proc.wait()!=0: raise RuntimeError('FFmpeg failed; see decode.log')
        finally:
            proc.stdout.close()
            if proc.poll() is None: proc.kill(); proc.wait()
        log.flush();log.seek(0);decode_text=log.read()
    checkpoint={'readings':readings,'ocr_seconds':ocr_seconds,'config':cfg,
                'video':str(args.video.resolve()),'source_size':args.video.stat().st_size,
                'source_mtime_ns':args.video.stat().st_mtime_ns}
    with (args.output/'raw-readings.json').open('w',encoding='utf8') as checkpoint_file:
        json.dump(checkpoint,checkpoint_file)
    pts=[float(x) for x in re.findall(r'\bpts_time:([\d.eE+-]+)',decode_text)]
    if len(pts)!=len(readings): raise RuntimeError(f'PTS/readings mismatch: {len(pts)} vs {len(readings)}')
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
             'config':cfg,'requested_seconds':args.seconds,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'parser_sha256':hashlib.sha256((ROOT/'readings.py').read_bytes()).hexdigest(),
             'runtime':{'onnxruntime':ort.__version__,'numpy':np.__version__},
             'observations':len(readings),'valid':valid_count,'ocr_seconds':ocr_seconds,'wall_seconds':time.perf_counter()-started,
             'timestamp_basis':'selected decoded frame PTS; FFmpeg default input start normalization','provider':rec._session.get_providers(),
             'status':'unreviewed OCR; sequence flags do not classify live play or replay'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
