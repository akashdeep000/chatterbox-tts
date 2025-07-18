"""
Text processing utilities for TTS
"""
from typing import List
import pysbd
import re


def _merge_small_chunks(chunks_list: List[str], min_words: int, max_len: int, buffer_percent: float = 0.10) -> List[str]:
    merged_chunks = []
    i = 0
    while i < len(chunks_list):
        current_chunk = chunks_list[i]
        if len(current_chunk.split()) < min_words:
            merged = False
            # Try merging with previous chunk
            if merged_chunks:
                prev_chunk = merged_chunks[-1]
                combined_len = len(prev_chunk) + len(current_chunk) + 1
                if combined_len <= max_len * (1 + buffer_percent):
                    merged_chunks[-1] = prev_chunk + " " + current_chunk
                    merged = True
            # If not merged with previous, try merging with next chunk (if available)
            if not merged and i < len(chunks_list) - 1:
                next_chunk = chunks_list[i+1]
                combined_len = len(current_chunk) + len(next_chunk) + 1
                if combined_len <= max_len * (1 + buffer_percent):
                    merged_chunks.append(current_chunk + " " + next_chunk)
                    i += 1 # Skip next chunk as it's merged
                    merged = True
            if not merged: # If still not merged, add as is (it's an unavoidable small chunk)
                merged_chunks.append(current_chunk)
        else:
            merged_chunks.append(current_chunk)
        i += 1
    return merged_chunks


def _split_oversized_segment(text: str, max_length: int) -> List[str]:
    """
    Breaks a long text string into smaller chunks, first by delimiters, then by words.

    This function performs several steps:
    1.  Splits the text by specified delimiters, keeping the delimiters attached
        to the preceding text. It handles cases with adjacent delimiters.
    2.  Processes each resulting phrase:
        - If a phrase is within `max_length`, it's kept as is.
        - If a phrase exceeds `max_length`, it's broken down by words.
    3.  When breaking by words, it ensures the last chunk is not a single word
        by merging it with the previous chunk if necessary.
    """
    # 1. Prioritize splitting by major delimiters
    major_delimiters = r'([;:])' # Semicolon, colon
    minor_delimiters = r'([,])' # Comma

    # Function to split by a given delimiter pattern and reconstruct phrases
    def split_and_reconstruct(text_to_split: str, delimiter_pattern: str) -> List[str]:
        parts = re.split(delimiter_pattern, text_to_split)
        reconstructed_phrases = []
        temp_phrase = ""
        for part in parts:
            if not part:
                continue
            temp_phrase += part
            if re.fullmatch(delimiter_pattern, part):
                if temp_phrase == part and reconstructed_phrases:
                    reconstructed_phrases[-1] += temp_phrase
                    temp_phrase = ""
                    continue
                reconstructed_phrases.append(temp_phrase.strip())
                temp_phrase = ""
        if temp_phrase.strip():
            reconstructed_phrases.append(temp_phrase.strip())
        return reconstructed_phrases

    # Initial split by major delimiters
    segments_by_major_delimiters = split_and_reconstruct(text, major_delimiters)

    intermediate_chunks = []
    for segment in segments_by_major_delimiters:
        if len(segment) <= max_length:
            intermediate_chunks.append(segment)
        else:
            # If still too long, split by minor delimiters
            segments_by_minor_delimiters = split_and_reconstruct(segment, minor_delimiters)
            for sub_segment in segments_by_minor_delimiters:
                if len(sub_segment) <= max_length:
                    intermediate_chunks.append(sub_segment)
                else:
                    # As a last resort, split by words
                    words = sub_segment.split()
                    current_word_chunk = ""
                    word_based_chunks = []
                    for word in words:
                        if len(current_word_chunk) + len(word) + 1 <= max_length:
                            current_word_chunk += (" " if current_word_chunk else "") + word
                        else:
                            if current_word_chunk:
                                word_based_chunks.append(current_word_chunk)
                            current_word_chunk = word
                    if current_word_chunk:
                        word_based_chunks.append(current_word_chunk)

                    # Apply merging immediately after word-based splitting
                    merged_word_chunks = _merge_small_chunks(word_based_chunks, 2, max_length)
                    intermediate_chunks.extend(merged_word_chunks)

    # Final merge pass for all intermediate chunks
    final_chunks = _merge_small_chunks(intermediate_chunks, 2, max_length)

    return [chunk.strip() for chunk in final_chunks if chunk.strip()]


from .dependencies import get_segmenter

def split_text_into_chunks(text: str, max_length: int = None) -> list:
    """
    Normalizes and splits text into manageable chunks for TTS processing.

    This function performs several steps:
    1.  Normalizes punctuation (e.g., smart quotes to straight quotes).
    2.  Uses `pysbd` to segment the text into sentences.
    3.  Capitalizes the beginning of the text.
    4.  Ensures each sentence has a proper ending punctuation mark.
    5.  If max_length exists and a chunk size is larger than max_length,
        it's broken down further using delimiters or by words.
        A special rule prevents creating a final chunk with only one word.
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

    # 2. Capitalize the very first letter of the whole text
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    if max_length is None:
        segmenter = get_segmenter()
        sentences = segmenter.segment(text)
        result = []
        sentence_enders = {".", "!", "?", "-"}
        for s in sentences:
            s = s.strip()
            if s:
                if not any(s.endswith(p) for p in sentence_enders):
                    s += "."
                result.append(s)
        return result

    # 3. Sentence Segmentation
    segmenter = get_segmenter()
    sentences = segmenter.segment(text)

    chunks = []
    current_chunk = ""
    sentence_enders = {".", "!", "?", "-"}

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 4. Ensure each sentence has a proper ending.
        if not any(sentence.endswith(p) for p in sentence_enders):
            sentence += "."

        # 5. Handle sentences that are too long from the start
        if len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            sub_chunks = _split_oversized_segment(sentence, max_length)
            chunks.extend(sub_chunks)
            continue

        # 6. Standard chunking logic
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    # Final merging pass to catch any remaining single-word chunks
    final_chunks = _merge_small_chunks(chunks, 2, max_length)
    return [chunk.strip() for chunk in final_chunks if chunk.strip()]

# Code for testing
# if __name__ == "__main__":
#     test_cases = [
#         {
#             "name": "Marvin's Secret (Original Issue)",
#             "text": """In a small coastal town where the waves whispered secrets to the shore, lived a cat named Marvin. Marvin wasn’t your ordinary cat—he wore round spectacles (don’t ask how they stayed on) and spent most of his afternoons reading newspapers in the town library’s window.

# Everyone assumed Marvin was just a quirky feline with a taste for sunlight and paper. But Marvin had a secret: he could understand everything. Human language, emotions, even Morse code if someone bothered to tap it out on a table.

# One foggy Tuesday morning, the mayor’s prized golden spoon vanished. The town spiraled into chaos. Posters went up. Rumors flew. The baker blamed the butcher, the butcher blamed the librarian, and the librarian just cried into her tea.

# Marvin watched it all from his sunny perch.

# That night, under the glow of a lamppost and the haunting call of a distant foghorn, Marvin slipped out. He darted across rooftops, slinked through hedges, and eventually found himself at the mayor’s house. There, buried under a pile of dirty laundry in the backyard, was the spoon—gleaming faintly beneath a sock with a hole in it.

# Standing above it, gnawing on a sausage, was Rollo the raccoon.

# “Seriously?” Marvin meowed.

# Rollo shrugged. “Shiny. Smelled like soup. You understand.”

# Marvin sighed. With a few quick leaps, he retrieved the spoon, dropped it on the mayor’s porch, rang the bell with his paw, and vanished into the night.

# The next day, the whole town buzzed about the mysterious return. No one knew who solved the mystery. But Marvin—back at his library perch, glasses askew, tail flicking—just turned the page of the newspaper with a quiet purr.

# Because some cats don’t chase mice.

# Some cats solve mysteries.""",
#             "max_length": 80
#         },
        # {
        #     "name": "Long Sentence with Multiple Delimiters",
        #     "text": "This is a very long sentence; it has multiple clauses, and it needs to be split carefully: by semicolons, by commas, and then by words if absolutely necessary.",
        #     "max_length": 30
        # },
        # {
        #     "name": "Sentence Ending with Single Word Chunk",
        #     "text": "This is a test. The last word is problematic.",
        #     "max_length": 20
        # },
        # {
        #     "name": "Short Text",
        #     "text": "Hello world.",
        #     "max_length": 100
        # },
        # {
        #     "name": "Empty Text",
        #     "text": "",
        #     "max_length": 50
        # },
        # {
        #     "name": "Very Long Continuous Text",
        #     "text": "This is an extremely long piece of text that contains no natural sentence breaks or delimiters, forcing the chunking algorithm to rely solely on word-based splitting. It is designed to test the robustness of the word-level splitting and the merging logic, ensuring that no single words are left isolated at the end of chunks, and that the buffer for merging is correctly applied. This text will go on for a very, very long time, stretching beyond any reasonable maximum length to truly challenge the function's ability to break down and reassemble content while maintaining readability and preventing orphaned words. The goal is to see how it handles continuous prose without punctuation cues, relying purely on the specified maximum length and the internal merging rules. This should provide a comprehensive test of the edge cases related to word-based chunking and the prevention of single-word remnants. We are adding more and more words to make sure that the chunking logic is truly robust and can handle any length of input without breaking down or producing undesirable output. The more words, the better the test, as it will expose any weaknesses in the algorithm's ability to manage very long strings of text. This continuous stream of words will push the limits of the function, ensuring that it performs as expected under extreme conditions. We will keep adding words until we are absolutely certain that the function is rock-solid and can handle anything thrown at it. This is a crucial test for the stability and reliability of the text processing utility. The continuous nature of this text, devoid of typical sentence or phrase delimiters, makes it an ideal candidate for stress-testing the word-level splitting and the subsequent merging operations. It forces the algorithm to make decisions purely based on length constraints, which is where the single-word prevention logic becomes most critical. We want to ensure that even in the absence of punctuation, the output chunks are meaningful and do not end abruptly with a lone word. This extensive text serves as a benchmark for the function's performance under challenging conditions, confirming its ability to deliver clean and well-formed chunks regardless of the input's structure. The more words we add, the more confident we become in the solution's resilience and accuracy. This is a truly exhaustive test, designed to leave no stone unturned in evaluating the chunking mechanism. We are committed to making this function as robust as possible, and this very long text is a key part of that commitment. It's a marathon, not a sprint, for our chunking algorithm, and we expect it to cross the finish line flawlessly, delivering perfectly formed chunks every time. This final addition of text is to ensure that the function can handle an extremely large input without any issues, confirming its scalability and efficiency. The goal is to have a text that is long enough to trigger all possible edge cases related to length constraints and word-based splitting. This will be the ultimate test of the function's capabilities.",
        #     "max_length": 70
        # }
    # ]

    # for test_case in test_cases:
    #     print("=" * 50)
    #     print(f"Test Case: {test_case['name']}")
    #     print(f"Max Length: {test_case['max_length']}")
    #     print("=" * 50)
    #     chunks = split_text_into_chunks(test_case['text'], max_length=test_case['max_length'])
    #     for i, chunk in enumerate(chunks):
    #         print(f"Chunk {i+1} (Length: {len(chunk)}):\n\"{chunk}\"\n")
    #     print("-" * 50)
    #     print(f"Total chunks: {len(chunks)}")
    #     print("-" * 50)
    #     print("\n")