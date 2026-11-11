"""Run ScoreSight OCR and align the result with NHL events and shifts."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from .align import align
from .nhl import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--game', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    bundle = load_bundle(args.game)
    subprocess.run([sys.executable, '-m', 'studio.scoreboard.extract', '--video', str(args.video),
                    '--config', str(args.config), '--output', str(args.output), '--ffmpeg', args.ffmpeg], check=True)
    with (args.output / 'observations.csv').open(newline='', encoding='utf8') as file:
        observations = list(csv.DictReader(file))
    summary = json.loads((args.output / 'summary.json').read_text(encoding='utf8'))
    result = align(observations, bundle['events'], bundle['shifts'], playoff=str(args.game)[4:6] == '03',
                   duration=summary['duration_seconds'])
    result['context'] = bundle['context']
    result['unavailable_context'] = bundle['unavailable_context']
    result['nhl_game_id'] = args.game
    (args.output / 'nhl-timeline.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf8')


if __name__ == '__main__':
    main()
