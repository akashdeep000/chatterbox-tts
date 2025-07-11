"""
Text processing utilities for TTS
"""
from typing import List

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
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
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

def split_text_for_streaming(text: str, chunk_size: int = 25) -> List[str]:
    """
    Splits text into chunks by word boundaries for low-latency streaming.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
        else:
            if current_chunk:
                chunks.append(current_chunk)

            if len(word) > chunk_size:
                for i in range(0, len(word), chunk_size):
                    chunks.append(word[i:i + chunk_size])
                current_chunk = ""
            else:
                current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def split_text_into_chunks(text: str, max_length: int = None) -> list:
    """Split text into manageable chunks for TTS processing"""
    if max_length is None:
        max_length = Config.MAX_CHUNK_LENGTH

    if len(text) <= max_length:
        return [text]

    # Try to split at sentence boundaries first
    sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    chunks = []
    current_chunk = ""

    # Split into sentences
    sentences = []
    temp_text = text

    while temp_text:
        best_split = len(temp_text)
        best_ending = ""

        for ending in sentence_endings:
            pos = temp_text.find(ending)
            if pos != -1 and pos < best_split:
                best_split = pos + len(ending)
                best_ending = ending

        if best_split == len(temp_text):
            # No sentence ending found, take the rest
            sentences.append(temp_text)
            break
        else:
            sentences.append(temp_text[:best_split])
            temp_text = temp_text[best_split:]

    # Group sentences into chunks
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If single sentence is too long, split it further
            if len(sentence) > max_length:
                # Split at commas, semicolons, etc.
                sub_delimiters = [', ', '; ', ' - ', ' — ']
                sub_chunks = [sentence]

                for delimiter in sub_delimiters:
                    new_sub_chunks = []
                    for chunk in sub_chunks:
                        if len(chunk) <= max_length:
                            new_sub_chunks.append(chunk)
                        else:
                            parts = chunk.split(delimiter)
                            current_part = ""
                            for part in parts:
                                if len(current_part) + len(delimiter) + len(part) <= max_length:
                                    current_part += (delimiter if current_part else "") + part
                                else:
                                    if current_part:
                                        new_sub_chunks.append(current_part)
                                    current_part = part
                            if current_part:
                                new_sub_chunks.append(current_part)
                    sub_chunks = new_sub_chunks

                # Add sub-chunks
                for sub_chunk in sub_chunks:
                    if len(sub_chunk) <= max_length:
                        chunks.append(sub_chunk.strip())
                    else:
                        # Last resort: split by words
                        words = sub_chunk.split()
                        current_word_chunk = ""
                        for word in words:
                            if len(current_word_chunk) + len(word) + 1 <= max_length:
                                current_word_chunk += (" " if current_word_chunk else "") + word
                            else:
                                if current_word_chunk:
                                    chunks.append(current_word_chunk)
                                current_word_chunk = word
                        if current_word_chunk:
                            chunks.append(current_word_chunk)
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]

    return chunks