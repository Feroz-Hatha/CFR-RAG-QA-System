def group_words_into_columns(words, page_width, gutter_ratio=0.08):
    """
    pdfplumber words have x0, x1 coords. We do a simple 2-column heuristic:
    - compute midline + gutter; assign each word to left/right column
    - return two lists (left_words, right_words) preserving reading order
    """
    if not words:
        return [], []
    x_median = page_width / 2.0
    gutter = page_width * gutter_ratio
    left, right = [], []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        center = (w["x0"] + w["x1"]) / 2.0
        if center <= x_median - gutter/2:
            left.append(w)
        elif center >= x_median + gutter/2:
            right.append(w)
        else:
            # tie-break: assign by closer side
            if abs(center - (x_median - gutter/2)) < abs(center - (x_median + gutter/2)):
                left.append(w)
            else:
                right.append(w)
    return left, right

def lines_from_words(words, line_tol=3.0):
    """
    Collapse words into lines by similar 'top' coordinate.
    """
    lines = []
    cur_y = None
    cur_line = []
    for w in words:
        if cur_y is None or abs(w["top"] - cur_y) <= line_tol:
            cur_line.append(w["text"])
            cur_y = w["top"] if cur_y is None else (cur_y + w["top"]) / 2.0
        else:
            if cur_line:
                lines.append(" ".join(cur_line))
            cur_line = [w["text"]]
            cur_y = w["top"]
    if cur_line:
        lines.append(" ".join(cur_line))
    return lines