document.addEventListener('DOMContentLoaded', () => {
    const apiKeyInput = document.getElementById('api-key');
    const saveApiKeyButton = document.getElementById('save-api-key');
    const baseUrlInput = document.getElementById('base-url');
    const saveBaseUrlButton = document.getElementById('save-base-url');
    const voiceList = document.getElementById('voice-list');
    const voiceFileInput = document.getElementById('voice-file');
    const uploadVoiceButton = document.getElementById('upload-voice');
    const ttsTextInput = document.getElementById('tts-text');
    const ttsVoiceSelect = document.getElementById('tts-voice');
    const generateTtsButton = document.getElementById('generate-tts');
    const stopTtsButton = document.getElementById('stop-tts');
    const ttsAudio = document.getElementById('tts-audio');
    const messageContainer = document.createElement('div');
    document.body.insertBefore(messageContainer, document.querySelector('.container'));
    const streamingLog = document.getElementById('streaming-log');

    // TTS Settings elements
    const textProcessingChunkSizeInput = document.getElementById('text-processing-chunk-size');
    const audioTokensPerSliceInput = document.getElementById('audio-tokens-per-slice');
    const removeLeadingMillisecondsInput = document.getElementById('remove-leading-milliseconds');
    const removeTrailingMillisecondsInput = document.getElementById('remove-trailing-milliseconds');
    const chunkOverlapStrategySelect = document.getElementById('chunk-overlap-strategy');
    const crossfadeDurationMillisecondsInput = document.getElementById('crossfade-duration-milliseconds');
    const saveTtsSettingsButton = document.getElementById('save-tts-settings');

    let apiKey = localStorage.getItem('apiKey');
    if (apiKey) {
        apiKeyInput.value = apiKey;
    }

    let baseUrl = localStorage.getItem('baseUrl') || window.location.origin;
    if (baseUrl) {
        baseUrlInput.value = baseUrl;
    }

    // Load TTS settings from localStorage
    const ttsSettings = JSON.parse(localStorage.getItem('ttsSettings')) || {};
    textProcessingChunkSizeInput.value = ttsSettings.text_processing_chunk_size || 100;
    audioTokensPerSliceInput.value = ttsSettings.audio_tokens_per_slice || 35;
    removeLeadingMillisecondsInput.value = ttsSettings.remove_leading_milliseconds || 0;
    removeTrailingMillisecondsInput.value = ttsSettings.remove_trailing_milliseconds || 0;
    chunkOverlapStrategySelect.value = ttsSettings.chunk_overlap_strategy || 'full';
    crossfadeDurationMillisecondsInput.value = ttsSettings.crossfade_duration_milliseconds || 8;

    let abortController;

    function showMessage(message, type = 'info') {
        messageContainer.textContent = message;
        messageContainer.className = `message ${type}`;
        setTimeout(() => {
            messageContainer.textContent = '';
            messageContainer.className = 'message';
        }, 3000);
    }

    saveApiKeyButton.addEventListener('click', () => {
        apiKey = apiKeyInput.value;
        localStorage.setItem('apiKey', apiKey);
        showMessage('API key saved!', 'success');
        loadVoices();
    });

    saveBaseUrlButton.addEventListener('click', () => {
        baseUrl = baseUrlInput.value;
        localStorage.setItem('baseUrl', baseUrl);
        showMessage('Base URL saved!', 'success');
        loadVoices();
    });

    async function loadVoices() {
        if (!apiKey) return;
        try {
            const response = await fetch(`${baseUrl}/voices`, {
                headers: { 'X-API-Key': apiKey }
            });
            if (!response.ok) {
                throw new Error(`Failed to fetch voices: ${response.statusText}`);
            }
            const voices = await response.json();
            voiceList.innerHTML = '';
            ttsVoiceSelect.innerHTML = '';
            voices.forEach(voice => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span>${voice}</span>
                    <div>
                        <button class="delete-voice" data-voice="${voice}">Delete</button>
                    </div>
                `;
                voiceList.appendChild(li);

                const option = document.createElement('option');
                option.value = voice;
                option.textContent = voice;
                ttsVoiceSelect.appendChild(option);
            });
        } catch (error) {
            console.error(error);
            showMessage(error.message, 'error');
        }
    }

    voiceList.addEventListener('click', (e) => {
        if (e.target.classList.contains('delete-voice')) {
            const voiceId = e.target.dataset.voice;
            if (confirm(`Are you sure you want to delete ${voiceId}?`)) {
                deleteVoice(voiceId);
            }
        }
    });

    async function deleteVoice(voiceId) {
        if (!apiKey) return;
        try {
            const response = await fetch(`${baseUrl}/voices/${voiceId}`, {
                method: 'DELETE',
                headers: { 'X-API-Key': apiKey }
            });
            if (!response.ok) {
                throw new Error(`Failed to delete voice: ${response.statusText}`);
            }
            showMessage('Voice deleted!', 'success');
            loadVoices();
        } catch (error) {
            console.error(error);
            showMessage(error.message, 'error');
        }
    }

    uploadVoiceButton.addEventListener('click', async () => {
        if (!apiKey || !voiceFileInput.files[0]) return;
        const formData = new FormData();
        formData.append('file', voiceFileInput.files[0]);
        try {
            uploadVoiceButton.textContent = 'Uploading...';
            uploadVoiceButton.disabled = true;
            const response = await fetch(`${baseUrl}/voices`, {
                method: 'POST',
                headers: { 'X-API-Key': apiKey },
                body: formData
            });
            if (!response.ok) {
                throw new Error(`Failed to upload voice: ${response.statusText}`);
            }
            showMessage('Voice uploaded!', 'success');
            loadVoices();
        } catch (error) {
            console.error(error);
            showMessage(error.message, 'error');
        } finally {
            uploadVoiceButton.textContent = 'Upload';
            uploadVoiceButton.disabled = false;
        }
    });

    function resetTtsControls() {
        generateTtsButton.textContent = 'Generate';
        generateTtsButton.disabled = false;
        stopTtsButton.disabled = true;
    }

    function generateTtsRest() {
        if (!apiKey || !ttsTextInput.value || !ttsVoiceSelect.value) return;

        generateTtsButton.textContent = 'Generating...';
        generateTtsButton.disabled = true;
        stopTtsButton.disabled = false;
        streamingLog.innerHTML = '';

        const url = new URL('/tts/generate', baseUrl);
        url.searchParams.append('text', ttsTextInput.value);
        url.searchParams.append('voice_id', ttsVoiceSelect.value);
        url.searchParams.append('api_key', apiKey);

        // Append TTS settings to URL
        url.searchParams.append('text_processing_chunk_size', textProcessingChunkSizeInput.value);
        url.searchParams.append('audio_tokens_per_slice', audioTokensPerSliceInput.value);
        url.searchParams.append('remove_leading_milliseconds', removeLeadingMillisecondsInput.value);
        url.searchParams.append('remove_trailing_milliseconds', removeTrailingMillisecondsInput.value);
        url.searchParams.append('chunk_overlap_strategy', chunkOverlapStrategySelect.value);
        url.searchParams.append('crossfade_duration_milliseconds', crossfadeDurationMillisecondsInput.value);

        ttsAudio.src = url.toString();
        ttsAudio.play();

        ttsAudio.onended = () => {
            resetTtsControls();
        };

        ttsAudio.onerror = () => {
            showMessage('Failed to play audio.', 'error');
            resetTtsControls();
        };
    }

    generateTtsButton.addEventListener('click', () => {
        generateTtsRest();
    });

    stopTtsButton.addEventListener('click', () => {
        ttsAudio.pause();
        ttsAudio.src = ""; // Stop the stream
        resetTtsControls();
    });

    if (apiKey) {
        loadVoices();
    }

    saveTtsSettingsButton.addEventListener('click', () => {
        const currentTtsSettings = {
            text_processing_chunk_size: parseInt(textProcessingChunkSizeInput.value),
            audio_tokens_per_slice: parseInt(audioTokensPerSliceInput.value),
            remove_leading_milliseconds: parseInt(removeLeadingMillisecondsInput.value),
            remove_trailing_milliseconds: parseInt(removeTrailingMillisecondsInput.value),
            chunk_overlap_strategy: chunkOverlapStrategySelect.value,
            crossfade_duration_milliseconds: parseInt(crossfadeDurationMillisecondsInput.value)
        };
        localStorage.setItem('ttsSettings', JSON.stringify(currentTtsSettings));
        showMessage('TTS settings saved!', 'success');
    });
});