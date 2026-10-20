"""Week 4-compatible tokenization. Reuse the original Flickr vocabulary unchanged."""
import html
import re


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.!?,;:]",
                      html.unescape(re.sub(r"<[^>]+>", " ", text)).lower())


def encode(text, words, length, caption=False):
    ids = [words.get(t, 1) for t in tokenize(text)]
    return [2] + ids[:length-2] + [3] if caption else ids[:length] or [1]
