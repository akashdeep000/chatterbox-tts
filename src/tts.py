import io
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
from .config import settings
from .voice_manager import VoiceManager
import time
from typing import Optional
import numpy as np
import torch.nn.functional as F
from chatterbox.models.s3tokenizer import drop_invalid_tokens
import perth

class TextToSpeechEngine:
    def __init__(self):
        print("Initializing TTS Engine...")
        device = settings.MODEL_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA is not available, falling back to CPU.")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS is not available, falling back to CPU.")
            device = "cpu"
        self.device = device
        self.tts = ChatterboxTTS.from_local(
            settings.MODEL_PATH,
            device=device
        )
        self.voice_manager = VoiceManager()
        self.watermarker = perth.PerthImplicitWatermarker()
        print("TTS Engine Initialized.")

    def _punc_norm(self, text: str) -> str:
        if len(text) == 0:
            return "You need to add some text for me to talk."
        if text[0].islower():
            text = text[0].upper() + text[1:]
        text = " ".join(text.split())
        punc_to_replace = [
            ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
            ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""),
            ("‘", "'"), ("’", "'"),
        ]
        for old, new in punc_to_replace:
            text = text.replace(old, new)
        if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
            text += "."
        return text

    def stream(self, text: str, voice_id: str = None):
        print(f"Streaming audio for text: '{text}'")
        audio_prompt_path = self.voice_manager.get_voice_path(voice_id) if voice_id else None

        header = self._create_wav_header(self.tts.sr)
        yield header

        for audio_chunk, _ in self._generate_stream(text, audio_prompt_path):
            # Convert the torch tensor to a numpy array, then to bytes (signed 16-bit PCM)
            # Ensure the audio_chunk is on CPU and converted to the correct data type
            audio_np = audio_chunk.cpu().squeeze(0).numpy()
            # Convert to signed 16-bit integers
            audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
            yield audio_bytes

    def _generate_stream(self, text: str, audio_prompt_path: Optional[str] = None, **kwargs):
        if audio_prompt_path:
            self.tts.prepare_conditionals(audio_prompt_path)

        text = self._punc_norm(text)
        text_tokens = self.tts.tokenizer.text_to_tokens(text).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        sot, eot = self.tts.t3.hp.start_text_token, self.tts.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        all_tokens_processed = []
        with torch.inference_mode():
            for token_chunk in self._inference_stream(t3_cond=self.tts.conds.t3, text_tokens=text_tokens, **kwargs):
                token_chunk = token_chunk[0]
                audio_tensor, _, success = self._process_token_buffer(
                    [token_chunk], all_tokens_processed, 50, fade_duration=0.05
                )
                if success:
                    yield audio_tensor, None

                if len(all_tokens_processed) == 0:
                    all_tokens_processed = token_chunk
                else:
                    all_tokens_processed = torch.cat([all_tokens_processed, token_chunk], dim=-1)

    def _inference_stream(self, t3_cond, text_tokens, max_new_tokens=1000, temperature=0.8, cfg_weight=0.5, chunk_size=25):
        from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor

        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
        initial_speech_tokens = self.tts.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, len_cond = self.tts.t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=initial_speech_tokens)

        if not getattr(self.tts.t3, 'compiled', False):
            from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
            from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
            alignment_analyzer = AlignmentStreamAnalyzer(self.tts.t3.tfmr, None, (len_cond, len_cond + text_tokens.size(-1)), 9, self.tts.t3.hp.stop_speech_token)
            self.tts.t3.patched_model = T3HuggingfaceBackend(
                self.tts.t3.cfg, self.tts.t3.tfmr,
                speech_enc=self.tts.t3.speech_emb, speech_head=self.tts.t3.speech_head
            )
            self.tts.t3.compiled = True

        bos_token = torch.tensor([[self.tts.t3.hp.start_speech_token]], dtype=torch.long, device=self.device)
        bos_embed = self.tts.t3.speech_emb(bos_token) + self.tts.t3.speech_pos_emb.get_fixed_embedding(0)
        inputs_embeds = torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1)

        generated_ids = bos_token.clone()
        chunk_buffer = []
        top_p_warper = TopPLogitsWarper(top_p=0.8)
        rep_penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)

        output = self.tts.t3.patched_model(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True)
        past = output.past_key_values

        for i in range(max_new_tokens):
            logits = output.logits[:, -1, :]
            logits_cond, logits_uncond = logits[0:1], logits[1:2]
            logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)
            if temperature != 1.0: logits /= temperature

            logits = rep_penalty_proc(generated_ids, logits.squeeze(1))
            logits = top_p_warper(None, logits)
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)

            chunk_buffer.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.view(-1) == self.tts.t3.hp.stop_speech_token:
                if chunk_buffer: yield torch.cat(chunk_buffer, dim=1)
                break

            if len(chunk_buffer) >= chunk_size:
                yield torch.cat(chunk_buffer, dim=1)
                chunk_buffer = []

            next_token_embed = self.tts.t3.speech_emb(next_token) + self.tts.t3.speech_pos_emb.get_fixed_embedding(i + 1)
            output = self.tts.t3.patched_model(inputs_embeds=torch.cat([next_token_embed, next_token_embed]), past_key_values=past, return_dict=True)
            past = output.past_key_values

    def _process_token_buffer(self, token_buffer, all_tokens_so_far, context_window, fade_duration=0.02):
        new_tokens = torch.cat(token_buffer, dim=-1)
        context_tokens = all_tokens_so_far[-context_window:] if len(all_tokens_so_far) > context_window else all_tokens_so_far
        new_tokens = torch.cat(token_buffer, dim=-1)
        context_tokens = all_tokens_so_far[-context_window:] if len(all_tokens_so_far) > context_window else all_tokens_so_far
        tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1) if len(all_tokens_so_far) > 0 else new_tokens

        clean_tokens = drop_invalid_tokens(tokens_to_process).to(self.device)
        if len(clean_tokens) == 0: return None, 0.0, False

        wav, _ = self.tts.s3gen.inference(speech_tokens=clean_tokens, ref_dict=self.tts.conds.gen)
        wav = wav.squeeze(0).detach().cpu().numpy()

        if len(context_tokens) > 0:
            samples_per_token = len(wav) / len(clean_tokens)
            skip_samples = int(len(context_tokens) * samples_per_token)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0: return None, 0.0, False

        fade_samples = int(fade_duration * self.tts.sr)
        if fade_samples > 0 and fade_samples < len(audio_chunk):
            audio_chunk[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)

        watermarked_chunk = self.watermarker.apply_watermark(audio_chunk, sample_rate=self.tts.sr)
        audio_tensor = torch.from_numpy(watermarked_chunk).unsqueeze(0)

        return audio_tensor, len(audio_chunk) / self.tts.sr, True

    def _create_wav_header(self, sample_rate, channels=1, sample_width=2, data_size=2**31-1-44):
        header = io.BytesIO()
        header.write(b'RIFF' + (data_size + 36).to_bytes(4, 'little') + b'WAVEfmt ' +
                     (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') +
                     channels.to_bytes(2, 'little') + sample_rate.to_bytes(4, 'little') +
                     (sample_rate * channels * sample_width).to_bytes(4, 'little') +
                     (channels * sample_width).to_bytes(2, 'little') +
                     (sample_width * 8).to_bytes(2, 'little') + b'data' +
                     data_size.to_bytes(4, 'little'))
        header.seek(0)
        return header.read()