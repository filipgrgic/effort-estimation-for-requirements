from config import MAX_CHARS


def chunk_text(text: str) -> list[str]:
    result = []
    if len(text) > MAX_CHARS:
        result = chunk_by_str(text, "\n\n")
    else:
        result.append(text)
    return result


def chunk_by_str(text: str, split_by: str) -> list[str]:
    chunks = [c for c in text.split(split_by) if c.strip()]

    compact = []
    i = 0
    size = len(chunks)
    while i < size:
        c = chunks[i]

        if len(c) > MAX_CHARS:
            if split_by == "\n\n":
                compact.extend(chunk_by_str(c, "\n"))
            elif split_by == "\n":
                compact.extend(chunk_by_str(c, ". "))
            else:
                hard_split = [c[i : i + MAX_CHARS] for i in range(0, len(c), MAX_CHARS)]
                compact.extend(hard_split)
            i += 1
            continue

        count = 1
        while (
            i + count < size
            and len(c) + len(split_by) + len(chunks[i + count]) <= MAX_CHARS
        ):
            c += split_by + chunks[i + count]
            count += 1
        compact.append(c)
        i += count

    return compact
