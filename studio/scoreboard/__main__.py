"""Run ScoreSight OCR and align the result with NHL events and shifts."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from .timeline import build_timeline
from .nhl import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Optional manual layout; automatic by default')
    parser.add_argument('--game', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    bundle = load_bundle(args.game)
    command = [sys.executable, '-m', 'studio.scoreboard.extract', '--video', str(args.video),
               '--output', str(args.output), '--ffmpeg', args.ffmpeg]
    if args.config:
        command += ['--config', str(args.config)]
    subprocess.run(command, check=True)
    with (args.output / 'observations.csv').open(newline='', encoding='utf8') as file:
        observations = list(csv.DictReader(file))
    summary = json.loads((args.output / 'summary.json').read_text(encoding='utf8'))
    result = build_timeline(observations, bundle, args.game, duration=summary['duration_seconds'])
    (args.output / 'nhl-timeline.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf8')


if __name__ == '__main__':
    main()
