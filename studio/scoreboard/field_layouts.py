"""Validated alternative clock/period crops shared by automatic and manual OCR."""
import json
from functools import lru_cache
from pathlib import Path

from .auto_layout import iou
from .readings import parse_reading

EMPTY = (('', 0), ('', 0))
CLOCK_CHARACTERS = '0123456789:.'
PERIOD_CHARACTERS = '0123456789STNDROILstndroil'


def profile_layouts(config):
    layouts = []
    for override in [{}] + config.get('variants', []):
        profile = {**config, **override}
        fields = [tuple(profile[key]) for key in ('clock', 'period')]
        for x, y, w, h in fields:
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > config['width'] or y + h > config['height']:
                raise ValueError('ROI falls outside configured video dimensions')
        if fields not in layouts:
            layouts.append(fields)
    return layouts


@lru_cache(maxsize=1)
def profiles():
    return [json.loads(p.read_text(encoding='utf8')) for p in sorted((Path(__file__).parent / 'profiles').glob('*.json'))]


def alternatives_for(fields, width, height):
    """Use calibrated variants only after detector geometry identifies their layout."""
    result = [fields]
    for profile in profiles():
        if abs(profile['width'] / profile['height'] - width / height) > .02:
            continue
        sx, sy = width / profile['width'], height / profile['height']
        layouts = [[(round(x*sx), round(y*sy), max(1,round(w*sx)), max(1,round(h*sy)))
                    for x,y,w,h in pair] for pair in profile_layouts(profile)]
        if not any(all(iou(a, b) >= .4 for a, b in zip(fields, pair)) for pair in layouts):
            continue
        for pair in layouts:
            if not any(all(iou(a, b) > .85 for a, b in zip(pair, existing)) for existing in result):
                result.append(pair)
    return result


def merge_readings(layouts, readings, threshold=.9):
    valid = []
    for i, fields in enumerate(layouts):
        pair = tuple(readings[2*i:2*i+2])
        confidence = min(r[1] or 0 for r in pair)
        clock, period, accepted = parse_reading(pair[0][0], pair[1][0], confidence, threshold)
        if accepted:
            valid.append((confidence, clock, period, pair, fields))
    if not valid or len({(v[1], v[2]) for v in valid}) != 1:
        return EMPTY, None
    best = max(valid, key=lambda v: v[0])
    return best[3], best[4]
