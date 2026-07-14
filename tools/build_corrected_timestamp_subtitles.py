import re
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUBTITLE = Path(r"C:\Users\ASUS\.codex\attachments\ffcd5f62-8580-49b5-8b72-ebf013940a11\pasted-text.txt")
CORRECTED_CHAPTERS = ROOT / "Midterm_Param" / "Corrected_YouTube_Transcript_Group11_Vietnamese.txt"
OUTPUT = ROOT / "Midterm_Param" / "Corrected_YouTube_Subtitles_Timestamp_Format.txt"


TIME_RE = re.compile(r"^(\d+:\d{2}:\d{2}\.\d{3}),(\d+:\d{2}:\d{2}\.\d{3})$")
CHAPTER_RE = re.compile(
    r"^Chapter\s+\d+:\s*(.*?)\n"
    r"([0-9:]+)\s*-\s*([0-9:]+)\n"
    r"=+\n\n"
    r"(.*?)(?=\n=+\nChapter\s+\d+:|\Z)",
    re.S | re.M,
)


def parse_seconds(t):
    parts = t.strip().split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(t)


def clean_chapter_text(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or set(line) == {"="}:
            continue
        if line.startswith("Topic:") or line.startswith("Speech Processing Project"):
            continue
        if line.startswith("Group 11") or line.startswith("Instructor:"):
            continue
        if line.startswith("Members:") or line.startswith("- "):
            continue
        if line.startswith("Ghi chú:"):
            continue
        lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_corrected_chapters():
    raw = CORRECTED_CHAPTERS.read_text(encoding="utf-8")
    chapters = []
    for match in CHAPTER_RE.finditer(raw):
        title, start, end, body = match.groups()
        chapters.append(
            {
                "title": title.strip(),
                "start": parse_seconds(start),
                "end": parse_seconds(end),
                "text": clean_chapter_text(body),
            }
        )
    if not chapters:
        raise RuntimeError("No corrected chapters parsed.")
    return chapters


def parse_source_blocks():
    raw = SOURCE_SUBTITLE.read_text(encoding="utf-8")
    chunks = re.split(r"\n\s*\n", raw.strip())
    blocks = []
    for chunk in chunks:
        lines = chunk.splitlines()
        if not lines:
            continue
        m = TIME_RE.match(lines[0].strip())
        if not m:
            continue
        start, end = m.groups()
        blocks.append(
            {
                "timestamp": lines[0].strip(),
                "start": parse_seconds(start),
                "end": parse_seconds(end),
            }
        )
    if not blocks:
        raise RuntimeError("No subtitle blocks parsed.")
    return blocks


def split_text_for_blocks(text, n):
    words = text.split()
    if n <= 0:
        return []
    if not words:
        return [""] * n
    chunks = []
    total = len(words)
    for i in range(n):
        a = round(i * total / n)
        b = round((i + 1) * total / n)
        part = " ".join(words[a:b]).strip()
        chunks.append(part)
    return chunks


def wrap_caption(text):
    if not text:
        return ""
    lines = textwrap.wrap(text, width=52, break_long_words=False, break_on_hyphens=False)
    if len(lines) <= 2:
        return "\n".join(lines)
    # Keep subtitle blocks compact, but do not drop meaning too aggressively.
    merged = [" ".join(lines[:2]), " ".join(lines[2:])]
    return "\n".join(textwrap.wrap(" ".join(merged), width=52, break_long_words=False, break_on_hyphens=False)[:3])


def main():
    chapters = parse_corrected_chapters()
    blocks = parse_source_blocks()

    output_blocks = []
    for chapter in chapters:
        chapter_blocks = [
            b
            for b in blocks
            if b["start"] >= chapter["start"] - 0.5 and b["start"] < chapter["end"] + 0.5
        ]
        parts = split_text_for_blocks(chapter["text"], len(chapter_blocks))
        for block, part in zip(chapter_blocks, parts):
            output_blocks.append(f"{block['timestamp']}\n{wrap_caption(part)}")

    used = {b.splitlines()[0] for b in output_blocks}
    for block in blocks:
        if block["timestamp"] not in used:
            output_blocks.append(f"{block['timestamp']}\n")

    OUTPUT.write_text("\n\n".join(output_blocks) + "\n", encoding="utf-8")
    print("Wrote", OUTPUT)
    print("Blocks:", len(output_blocks))


if __name__ == "__main__":
    main()
