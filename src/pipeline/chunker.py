from config import MAX_CHARS


def chunk_text(text: str) -> list[str]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    compact = []
    i = 0
    size = len(paragraphs)
    while i < size:
        p = paragraphs[i]

        if len(p) > MAX_CHARS:
            compact.extend(split_paragraph(p))
            i += 1
            continue

        c = 1
        while i + c < size and len(p) + 2 + len(paragraphs[i + c]) <= MAX_CHARS:
            p += "\n\n" + paragraphs[i + c]
            c += 1
        compact.append(p)
        i += c

    return compact


def split_paragraph(p: str) -> list[str]:
    # TODO
    return
