"""Parse scoreboard text without loading an OCR runtime."""
import re


def parse_reading(clock, period, confidence, threshold):
    text = clock.strip()
    if text.startswith(':'):
        text = text[1:] if re.fullmatch(r'\d[.]\d', text[1:]) else '0' + text
    match = re.fullmatch(r'(\d{1,2}):([0-5]\d(?:[.]\d)?)', text)
    # The ordinal suffix makes I/l versus 1 unambiguous in this field.
    periods = {'1ST':1, 'IST':1, 'LST':1, '2ND':2, '3RD':3, 'OT':4, 'SO':5}
    p = periods.get(period.strip().upper().replace(' ', ''))
    overtime = re.fullmatch(r'(\d{1,2})OT', period.strip().upper().replace(' ', ''))
    if overtime and 1 <= int(overtime[1]) <= 17:
        p = 3 + int(overtime[1])
    seconds = int(match[1])*60+float(match[2]) if match else None
    if seconds is None and re.fullmatch(r'[0-5]?\d[.]\d', text):
        seconds = float(text)
    valid = seconds is not None and seconds <= 1200 and p is not None and confidence >= threshold
    return seconds, p, valid

