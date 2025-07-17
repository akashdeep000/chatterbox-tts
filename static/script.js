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
    textProcessingChunkSizeInput.value = ttsSettings.text_processing_chunk_size || 150;
    audioTokensPerSliceInput.value = ttsSettings.audio_tokens_per_slice || 35;
    removeLeadingMillisecondsInput.value = ttsSettings.remove_leading_milliseconds || 0;
    removeTrailingMillisecondsInput.value = ttsSettings.remove_trailing_milliseconds || 0;
    chunkOverlapStrategySelect.value = ttsSettings.chunk_overlap_strategy || 'full';
    crossfadeDurationMillisecondsInput.value = ttsSettings.crossfade_duration_milliseconds || 30;

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

    async function generateTts() {
        if (!apiKey || !ttsTextInput.value || !ttsVoiceSelect.value) return;

        generateTtsButton.textContent = 'Generating...';
        generateTtsButton.disabled = true;
        stopTtsButton.disabled = false;
        streamingLog.innerHTML = '';

        abortController = new AbortController();
        const signal = abortController.signal;

        const formatSelect = document.getElementById('format-select');
        const selectedFormat = formatSelect.value;
        const useMse = selectedFormat === 'fmp4';

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
        url.searchParams.append('format', selectedFormat);

        const headers = { 'X-API-Key': apiKey };
        if (useMse) {
            headers['Accept'] = 'audio/mp4';
        } else {
            // For non-MSE, the browser will set the Accept header,
            // but the 'format' URL param takes precedence on the server.
        }

        if (useMse) {
            const mediaSource = new MediaSource();
            ttsAudio.src = URL.createObjectURL(mediaSource);

            mediaSource.addEventListener('sourceopen', async () => {
                URL.revokeObjectURL(ttsAudio.src);
                const sourceBuffer = mediaSource.addSourceBuffer('audio/mp4; codecs="mp4a.40.2"');

                try {
                    const response = await fetch(url, { headers, signal });
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

                    const reader = response.body.getReader();

                    const appendNextChunk = async () => {
                        const { done, value } = await reader.read();
                        if (done) {
                            if (mediaSource.readyState === 'open') mediaSource.endOfStream();
                            resetTtsControls();
                            return;
                        }
                        sourceBuffer.appendBuffer(value);
                    };

                    sourceBuffer.addEventListener('updateend', appendNextChunk);
                    appendNextChunk(); // Start the process

                } catch (error) {
                    if (error.name !== 'AbortError') {
                        console.error('Error during TTS generation:', error);
                        showMessage('Failed to generate TTS.', 'error');
                    }
                    resetTtsControls();
                }
            });
        } else {
            // Direct playback for WAV and MP3
            try {
                // The abortController is not used here, but the stop button will still work
                // by pausing and resetting the src. The 'onerror' handler will catch errors.
                ttsAudio.src = url.toString();
            } catch (error) {
                console.error('Error setting audio source:', error);
                showMessage('Failed to generate TTS.', 'error');
                resetTtsControls();
            }
        }

        ttsAudio.play().catch(e => {
            console.error("Audio play failed:", e);
            resetTtsControls();
        });

        ttsAudio.onended = () => {
            resetTtsControls();
        };

        ttsAudio.onerror = (e) => {
            console.error("Audio element error:", e);
            showMessage('Failed to play audio.', 'error');
            resetTtsControls();
        };
    }

    generateTtsButton.addEventListener('click', generateTts);

    stopTtsButton.addEventListener('click', () => {
        if (abortController) {
            abortController.abort();
        }
        ttsAudio.pause();
        ttsAudio.src = "";
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

    // --- System Status ---
    const cpuUtilBar = document.getElementById('cpu-util-bar');
    const cpuUtilText = document.getElementById('cpu-util-text');
    const ramUsageBar = document.getElementById('ram-usage-bar');
    const ramUsageText = document.getElementById('ram-usage-text');
    const gpuUtilBar = document.getElementById('gpu-util-bar');
    const gpuUtilText = document.getElementById('gpu-util-text');
    const vramUsageBar = document.getElementById('vram-usage-bar');
    const vramUsageText = document.getElementById('vram-usage-text');
    const statusError = document.getElementById('status-error');

    async function updateSystemStatus() {
        if (!apiKey) {
            statusError.textContent = 'API Key not set. Cannot fetch status.';
            return;
        }
        try {
            const response = await fetch(`${baseUrl}/system-status`, {
                headers: { 'X-API-Key': apiKey }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`Failed to fetch status: ${errorData.detail || response.statusText}`);
            }

            const data = await response.json();
            statusError.textContent = ''; // Clear previous errors

            // Update CPU
            if (data.cpu && !data.cpu.error) {
                const cpuPercent = data.cpu.utilization_percent;
                cpuUtilBar.style.width = `${cpuPercent}%`;
                cpuUtilText.textContent = `${cpuPercent.toFixed(1)}%`;

                const ramPercent = data.cpu.ram_gb.percent_used;
                ramUsageBar.style.width = `${ramPercent}%`;
                ramUsageText.textContent = `${data.cpu.ram_gb.used} / ${data.cpu.ram_gb.total} GB (${ramPercent}%)`;
            } else {
                statusError.textContent = `CPU/RAM Error: ${data.cpu.error}`;
            }

            // Update GPU
            if (data.gpu && !data.gpu.error) {
                const gpuPercent = data.gpu.utilization_percent.gpu;
                gpuUtilBar.style.width = `${gpuPercent}%`;
                gpuUtilText.textContent = `${gpuPercent}%`;

                const vramTotal = data.gpu.memory_gb.total;
                const vramUsed = data.gpu.memory_gb.used;
                const vramPercent = vramTotal > 0 ? (vramUsed / vramTotal) * 100 : 0;
                vramUsageBar.style.width = `${vramPercent}%`;
                vramUsageText.textContent = `${vramUsed} / ${vramTotal} GB (${vramPercent.toFixed(1)}%)`;
            } else {
                // Don't overwrite CPU error if GPU is also unavailable
                if (!statusError.textContent) {
                    statusError.textContent = `GPU Error: ${data.gpu.error || data.gpu.reason}`;
                }
            }

        } catch (error) {
            console.error('Error fetching system status:', error);
            statusError.textContent = error.message;
        }
    }

    if (apiKey) {
        setInterval(updateSystemStatus, 2000); // Update every 2 seconds
        updateSystemStatus(); // Initial call
    }
});