"""Find broadcast clock/period fields with ScoreSight detection plus OCR validation."""
import math
import time

from .auto_layout import iou
from .readings import parse_reading
from .field_layouts import EMPTY, CLOCK_CHARACTERS, PERIOD_CHARACTERS, alternatives_for, merge_readings
from .runtime import batch_frames


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
    # A penalty countdown elsewhere on the board is not a clock/period pair.
    x, y, w, h = chosen[0][0]
    px, py, pw, ph = chosen[1][0]
    gap = math.hypot(max(0, x-px-pw, px-x-w), max(0, y-py-ph, py-y-h))
    if gap > 3 * max(h, ph):
        return None
    return [x[0] for x in chosen], tuple(x[1] for x in chosen)


class AutoReader:
    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer
        self.fields = None
        self.layouts = []
        self.batch_frames = batch_frames()
        self.until_search = 0
        self.layout_seconds = 0
        self.ocr_seconds = 0

    def _ocr(self, image, boxes, whitelists=None):
        start = time.perf_counter()
        crops = [image[y:y+h, x:x+w].copy() for x,y,w,h in boxes]
        result = self.recognizer.recognize(crops, whitelists) if whitelists else self.recognizer.recognize(crops)
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
            for i, alternative in zip(retry, self.recognizer.recognize(crops, [whitelists[i] for i in retry]) if whitelists else self.recognizer.recognize(crops)):
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

    def _remember(self, fields, image):
        known = alternatives_for(fields, image.shape[1], image.shape[0])
        for previous in self.layouts:
            if not any(all(iou(a,b) > .85 for a,b in zip(previous, pair)) for pair in known):
                known.append(previous)
        self.layouts = known[:8]
        self.fields = fields

    def _known(self, image):
        layouts = self.layouts or [self.fields]
        boxes = [box for pair in layouts for box in pair]
        allowed = [CLOCK_CHARACTERS, PERIOD_CHARACTERS] * len(layouts)
        return merge_readings(layouts, self._ocr(image, boxes, allowed))

    def read(self, image):
        self.until_search -= 1
        known = self._known(image) if self.fields else (EMPTY, None)
        if self.until_search > 0:
            if known[1]:
                self.fields = known[1]
            return known
        pair = self._search(image)
        self.until_search = 20 if pair or self.fields else 4
        if pair:
            fields, readings = pair
            if known[1] and iou(known[1][0], fields[0]) < .5:
                # Keep working game-clock geometry when a penalty timer appears.
                # Conflicting text remains unresolved, without poisoning the cache.
                return merge_readings([known[1], fields], list(known[0]) + list(readings))
            self._remember(fields, image)
            # Discovery identifies field roles; every accepted known-field read
            # thereafter uses role-specific decoding and checks all variants.
            discovered = self._known(image)
            if known[1] and discovered[1]:
                return merge_readings([known[1], discovered[1]], list(known[0]) + list(discovered[0]))
            return discovered
        return known

    def read_many(self, images):
        """Batch stable layouts, retaining the same ambiguity/recovery policy."""
        results = []
        index = 0
        while index < len(images):
            if not self.fields or self.until_search <= 1:
                results.append(self.read(images[index]))
                index += 1
                continue
            layouts = self.layouts or [self.fields]
            boxes = [box for pair in layouts for box in pair]
            count = min(len(images) - index, self.until_search - 1, self.batch_frames)
            crops = [image[y:y+h, x:x+w].copy()
                     for image in images[index:index+count] for x,y,w,h in boxes]
            allowed = [CLOCK_CHARACTERS, PERIOD_CHARACTERS] * (len(layouts) * count)
            start = time.perf_counter()
            batch = self.recognizer.recognize(crops, allowed)
            self.ocr_seconds += time.perf_counter() - start
            stride = len(boxes)
            for offset in range(count):
                result = merge_readings(layouts, batch[stride*offset:stride*(offset+1)])
                if not result[1]:
                    # Retry low-confidence period crops, never carry text forward.
                    results.append(self.read(images[index]))
                    index += 1
                    break
                self.until_search -= 1
                self.fields = result[1]
                results.append(result)
                index += 1
        return results
