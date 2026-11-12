"""Find broadcast clock/period fields with ScoreSight detection plus OCR validation."""
import math
import time

from .auto_layout import iou
from .readings import parse_reading

EMPTY = (('', 0), ('', 0))


def choose_pair(candidates):
    """Accept one spatial clock and one spatial period; reject competing clocks."""
    roles = [[], []]
    for rect, reading in candidates:
        text, confidence = reading
        if (confidence or 0) < .9:
            continue
        if parse_reading(text, '1ST', confidence, .9)[2]:
            roles[0].append((rect, reading))
        if parse_reading('20:00', text, confidence, .9)[2]:
            roles[1].append((rect, reading))
    chosen = []
    for choices in roles:
        if not choices:
            return None
        best = max(choices, key=lambda x: x[1][1])
        if any(iou(best[0], other[0]) < .5 for other in choices):
            return None
        chosen.append(best)
    if iou(chosen[0][0], chosen[1][0]) > .3:
        return None
    return [x[0] for x in chosen], tuple(x[1] for x in chosen)


class AutoReader:
    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer
        self.fields = None
        self.until_search = 0
        self.layout_seconds = 0
        self.ocr_seconds = 0

    def _ocr(self, image, boxes):
        start = time.perf_counter()
        result = self.recognizer.recognize([image[y:y+h, x:x+w].copy() for x,y,w,h in boxes])
        # Tight period glyphs can include the clock above in detector padding.
        # Retry a vertical inset only for plausible, low-confidence period text.
        retry = [i for i, (text, confidence) in enumerate(result)
                 if (confidence or 0) < .9 and parse_reading('20:00', text, 1, .9)[2]]
        if retry:
            crops = []
            for i in retry:
                x,y,w,h = boxes[i]
                pad = max(1, round(h * .15))
                crops.append(image[y+pad:y+h-pad, x:x+w].copy())
            for i, alternative in zip(retry, self.recognizer.recognize(crops)):
                # Reject conflicting readings rather than choosing confidence alone.
                if (parse_reading('20:00', alternative[0], 1, .9)[1]
                        == parse_reading('20:00', result[i][0], 1, .9)[1]
                        and (alternative[1] or 0) > (result[i][1] or 0)):
                    result[i] = alternative
        self.ocr_seconds += time.perf_counter()-start
        return result

    def _search(self, image):
        height, width = image.shape[:2]
        # Full picture first; quadrants make small broadcast overlays legible to the detector.
        tiles = [(0, 0, width, height)] + [(x, y, width//2, height//2)
                 for y in (0, height-height//2) for x in (0, width-width//2)]
        for tx, ty, tw, th in tiles:
            start = time.perf_counter()
            detections = self.detector.detect(image[ty:ty+th, tx:tx+tw], min_score=.3)
            self.layout_seconds += time.perf_counter()-start
            boards = [d for d in detections if d.kind == 'board' and d.score >= .5
                      and d.rect[2]*d.rect[3] <= width*height*.25]
            pairs = []
            for board in boards:
                bx, by, bw, bh = board.rect
                boxes = []
                for d in detections:
                    x,y,w,h = d.rect
                    if d.kind == 'board' or not (bx <= x+w/2 <= bx+bw and by <= y+h/2 <= by+bh):
                        continue
                    pad = .1*h
                    left, top = max(0, math.floor(x+tx-pad)), max(0, math.floor(y+ty-pad))
                    right, bottom = min(width, math.ceil(x+tx+w+pad)), min(height, math.ceil(y+ty+h+pad))
                    rect = (left, top, right-left, bottom-top)
                    if not any(iou(rect, other) > .7 for other in boxes):
                        boxes.append(rect)
                if not boxes:
                    continue
                pair = choose_pair(list(zip(boxes, self._ocr(image, boxes))))
                if pair:
                    pairs.append(pair)
            if pairs:
                first = pairs[0]
                if any(iou(first[0][0], pair[0][0]) < .5 for pair in pairs[1:]):
                    return None
                return first
        return None

    def read(self, image):
        self.until_search -= 1
        if self.fields:
            readings = self._ocr(image, self.fields)
            confidence = min((r[1] or 0 for r in readings), default=0)
            if parse_reading(readings[0][0], readings[1][0], confidence, .9)[2] and self.until_search > 0:
                return tuple(readings), self.fields
            if self.until_search > 0:
                # Remember geometry, never text. Missing/low-confidence samples
                # stay empty; later frames are read afresh at the known location.
                return EMPTY, None
        if self.until_search > 0:
            return EMPTY, None
        pair = self._search(image)
        self.until_search = 20 if pair or self.fields else 4
        if pair:
            self.fields, readings = pair
            return readings, self.fields
        return EMPTY, None

    def read_many(self, images):
        """Batch stable crops while preserving per-frame invalidation and search order."""
        results = []
        index = 0
        while index < len(images):
            if not self.fields or self.until_search <= 1:
                results.append(self.read(images[index]))
                index += 1
                continue
            fields = self.fields
            count = min(len(images) - index, self.until_search - 1, 16)
            crops = [image[y:y+h, x:x+w].copy()
                     for image in images[index:index+count] for x,y,w,h in fields]
            start = time.perf_counter()
            batch = self.recognizer.recognize(crops)
            self.ocr_seconds += time.perf_counter() - start
            for offset in range(count):
                readings = tuple(batch[2*offset:2*offset+2])
                confidence = min(r[1] or 0 for r in readings)
                if not parse_reading(readings[0][0], readings[1][0], confidence, .9)[2]:
                    # Discard speculative later crops: the board may have moved.
                    results.append(self.read(images[index]))
                    index += 1
                    break
                self.until_search -= 1
                results.append((readings, fields))
                index += 1
        return results
