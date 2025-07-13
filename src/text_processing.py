"""
Text processing utilities for TTS
"""
from typing import List
import pysbd
import re

def punc_norm(text: str) -> str:
    """
    Normalizes punctuation in the input text.
    """
    if len(text) == 0:
        return ""
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [
        ("...", ". "), ("…", ". "), (" - ", ", "),
        ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""),
        ("‘", "'"), ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text

def split_text_into_chunks(text: str, max_length: int = None) -> list:
    """Split text into manageable chunks for TTS processing using pysbd."""
    if max_length is None or len(text) <= max_length:
        return [text] if text.strip() else []

    seg = pysbd.Segmenter(language="en", clean=False)
    sentences = seg.segment(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)

            if len(sentence) > max_length:
                # This sentence is too long. We need to split it more granularly.
                # First, split by a comprehensive set of delimiters to get clauses or phrases.
                # The regex splits the string by the delimiters, keeping the delimiters in the list.
                delimiters = r'([,;:—–\-"\'“”‘’\(\)\[\]\{\}])'
                parts = re.split(delimiters, sentence)

                # Reconstruct phrases by joining each text part with its following delimiter.
                phrases = []
                current_phrase = ""
                for part in parts:
                    if not part:
                        continue
                    current_phrase += part
                    # If the part is a delimiter, we consider the phrase complete.
                    if re.fullmatch(delimiters, part):
                        phrases.append(current_phrase.strip())
                        current_phrase = ""
                if current_phrase.strip():
                    phrases.append(current_phrase.strip())

                # Now, process each smaller phrase.
                for phrase in phrases:
                    if len(phrase) <= max_length:
                        # This phrase is short enough, add it as a chunk.
                        chunks.append(phrase)
                    else:
                        # This phrase is STILL too long. As a last resort, split by words.
                        words = phrase.split()
                        word_chunk = ""
                        for word in words:
                            if len(word_chunk) + len(word) + 1 <= max_length:
                                word_chunk += (" " if word_chunk else "") + word
                            else:
                                if word_chunk:
                                    chunks.append(word_chunk)
                                word_chunk = word
                        if word_chunk:
                            chunks.append(word_chunk)
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return [chunk.strip() for chunk in chunks if chunk.strip()]