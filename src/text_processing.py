"""
Text processing utilities for TTS
"""
from typing import List
import pysbd
import re


def split_text_into_chunks(text: str, max_length: int = None) -> list:
    """
    Normalizes and splits text into manageable chunks for TTS processing.

    This function performs several steps:
    1.  Normalizes punctuation (e.g., smart quotes to straight quotes).
    3.  Uses `pysbd` to segment the text into sentences.
    4.  Ensures each sentence has a proper ending punctuation mark.
    5.  Chunks sentences together up to `max_length`.
    6.  If a single sentence exceeds `max_length`, it's broken down further
        into phrases or words, without adding extra punctuation to fragments.
    """
    # 1. Pre-normalization
    if not text or not text.strip():
        return []
    text = " ".join(text.split())
    punc_to_replace = [
        ("...", ". "), ("…", ". "), (" - ", ", "),
        ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""),
        ("‘", "'"), ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # If the whole text is one chunk, process and return it.
    if max_length is None or len(text) <= max_length:
        sentence_enders = {".", "!", "?", "-", ","}
        if not any(text.endswith(p) for p in sentence_enders):
            text += "."
        return [text] if text.strip() else []

    # 2. Sentence Segmentation
    seg = pysbd.Segmenter(language="en", clean=False)
    sentences = seg.segment(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Ensure each sentence from pysbd has a proper ending.
        # This handles cases where the segmenter might miss it.
        sentence_enders = {".", "!", "?", "-"}  # Note: Comma is not a sentence ender
        if not any(sentence.endswith(p) for p in sentence_enders):
            sentence += "."

        # 3. Chunking logic
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # 4. Handle sentences that are too long
            if len(sentence) > max_length:
                # The fragments created here should NOT get a period added.
                delimiters = r'([,;\)\]\}])'
                parts = re.split(delimiters, sentence)

                # Reconstruct phrases by joining each text part with its following delimiter.
                phrases = []
                temp_phrase = ""
                for part in parts:
                    if not part:
                        continue
                    temp_phrase += part
                    if re.fullmatch(delimiters, part):
                        phrases.append(temp_phrase.strip())
                        temp_phrase = ""
                if temp_phrase.strip():
                    phrases.append(temp_phrase.strip())

                # Process each smaller phrase.
                for phrase in phrases:
                    if len(phrase) <= max_length:
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

# Code for testing
if __name__ == "__main__":
    text = """In a small coastal town where the waves whispered secrets to the shore, lived a cat named Marvin. Marvin wasn’t your ordinary cat—he wore round spectacles (don’t ask how they stayed on) and spent most of his afternoons reading newspapers in the town library’s window.

Everyone assumed Marvin was just a quirky feline with a taste for sunlight and paper. But Marvin had a secret: he could understand everything. Human language, emotions, even Morse code if someone bothered to tap it out on a table.

One foggy Tuesday morning, the mayor’s prized golden spoon vanished. The town spiraled into chaos. Posters went up. Rumors flew. The baker blamed the butcher, the butcher blamed the librarian, and the librarian just cried into her tea.

Marvin watched it all from his sunny perch.

That night, under the glow of a lamppost and the haunting call of a distant foghorn, Marvin slipped out. He darted across rooftops, slinked through hedges, and eventually found himself at the mayor’s house. There, buried under a pile of dirty laundry in the backyard, was the spoon—gleaming faintly beneath a sock with a hole in it.

Standing above it, gnawing on a sausage, was Rollo the raccoon.

“Seriously?” Marvin meowed.

Rollo shrugged. “Shiny. Smelled like soup. You understand.”

Marvin sighed. With a few quick leaps, he retrieved the spoon, dropped it on the mayor’s porch, rang the bell with his paw, and vanished into the night.

The next day, the whole town buzzed about the mysterious return. No one knew who solved the mystery. But Marvin—back at his library perch, glasses askew, tail flicking—just turned the page of the newspaper with a quiet purr.

Because some cats don’t chase mice.

Some cats solve mysteries."""
    max_len = 75
    print("-" * 50)
    print(f"Splitting text with max_length = {max_len}")
    print("-" * 50)
    chunks = split_text_into_chunks(text, max_length=max_len)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (Length: {len(chunk)}):\n\"{chunk}\"\n")
    print("-" * 50)
    print(f"Total chunks: {len(chunks)}")
    print("-" * 50)