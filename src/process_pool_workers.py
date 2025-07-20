"""
Worker functions for the ProcessPoolExecutor to avoid GIL bottlenecks and pickling issues.
"""
from chatterbox.models.tokenizers import EnTokenizer

# A global variable to hold the tokenizer instance within each worker process.
tokenizer_instance = None

def initialize_tokenizer(model_path: str):
    """
    This function is called once per worker process to initialize the tokenizer.
    """
    global tokenizer_instance
    # Each process creates its own instance, avoiding the need to pickle the object.
    tokenizer_instance = EnTokenizer(model_path)

def tokenize_chunk_worker(text_chunk: str) -> list:
    """
    The actual worker function that performs tokenization.
    It returns a simple list of integers, which is safe to pickle.
    """
    if tokenizer_instance is None:
        raise RuntimeError("Tokenizer not initialized in this process. Call initialize_tokenizer first.")
    return tokenizer_instance.text_to_tokens(text_chunk).tolist()