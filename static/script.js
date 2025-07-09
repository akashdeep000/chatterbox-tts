document.addEventListener('DOMContentLoaded', () => {
    const apiKeyInput = document.getElementById('api-key');
    const saveApiKeyButton = document.getElementById('save-api-key');
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

    let apiKey = localStorage.getItem('apiKey');
    if (apiKey) {
        apiKeyInput.value = apiKey;
    }

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

    async function loadVoices() {
        if (!apiKey) return;
        try {
            const response = await fetch('/voices', {
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
            const response = await fetch(`/voices/${voiceId}`, {
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
            const response = await fetch('/voices', {
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

    async function generateTtsRest() {
        if (!apiKey || !ttsTextInput.value || !ttsVoiceSelect.value) return;

        abortController = new AbortController();
        generateTtsButton.textContent = 'Generating...';
        generateTtsButton.disabled = true;
        stopTtsButton.disabled = false;
        streamingLog.innerHTML = '';

        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        let audioQueue = [];
        let isPlaying = false;
        let wavHeader = null;
        const MIN_BUFFER_SIZE = 24000; // Buffer at least this many bytes before playing
        let dataBuffer = new Uint8Array(0);
        let allChunks = [];

        function playFromQueue() {
            if (isPlaying || audioQueue.length === 0) return;

            isPlaying = true;
            const bufferToPlay = audioQueue.shift();
            const source = audioContext.createBufferSource();
            source.buffer = bufferToPlay;
            source.connect(audioContext.destination);
            source.onended = () => {
                isPlaying = false;
                playFromQueue();
            };
            source.start();
        }

        try {
            const response = await fetch('/tts/generate', {
                method: 'POST',
                headers: {
                    'X-API-Key': apiKey,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: ttsTextInput.value,
                    voice_id: ttsVoiceSelect.value
                }),
                signal: abortController.signal
            });

            if (!response.ok) {
                throw new Error(`Failed to generate TTS: ${response.statusText}`);
            }

            const reader = response.body.getReader();

            // Read header
            const { value: headerChunk, done: headerDone } = await reader.read();
            if (headerDone || headerChunk.length !== 44) {
                console.error("Stream did not start with a 44-byte WAV header.");
                return;
            }
            wavHeader = headerChunk;

            while (true) {
                const { done, value: newChunk } = await reader.read();

                if (newChunk) {
                    allChunks.push(newChunk);
                    // Append new data to our main buffer
                    const newBuffer = new Uint8Array(dataBuffer.length + newChunk.length);
                    newBuffer.set(dataBuffer, 0);
                    newBuffer.set(newChunk, dataBuffer.length);
                    dataBuffer = newBuffer;
                }

                // If we have enough data in the buffer, or if the stream is done, process a segment
                if (dataBuffer.length >= MIN_BUFFER_SIZE || (done && dataBuffer.length > 0)) {
                    const segmentToProcess = dataBuffer;
                    dataBuffer = new Uint8Array(0); // Clear the buffer

                    const logEntry = document.createElement('p');
                    logEntry.textContent = `Processing segment of size: ${segmentToProcess.length}`;
                    streamingLog.appendChild(logEntry);

                    // Create a valid WAV file for this segment
                    const wavFileSegment = new ArrayBuffer(wavHeader.length + segmentToProcess.length);
                    const view = new Uint8Array(wavFileSegment);
                    view.set(wavHeader, 0);
                    view.set(segmentToProcess, wavHeader.length);

                    const dataView = new DataView(wavFileSegment);
                    dataView.setUint32(4, 36 + segmentToProcess.length, true);
                    dataView.setUint32(40, segmentToProcess.length, true);

                    try {
                        const decodedBuffer = await audioContext.decodeAudioData(wavFileSegment);
                        audioQueue.push(decodedBuffer);
                        playFromQueue();
                    } catch (e) {
                        console.error("Error decoding audio segment", e);
                        break;
                    }
                }

                if (done) {
                    const blob = new Blob([wavHeader, ...allChunks], { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(blob);
                    ttsAudio.src = audioUrl;
                    resetTtsControls();
                    break;
                }
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error(error);
                showMessage(error.message, 'error');
            }
            resetTtsControls();
        }
    }

    generateTtsButton.addEventListener('click', () => {
        generateTtsRest();
    });

    stopTtsButton.addEventListener('click', () => {
        if (abortController) {
            abortController.abort();
        }
    });

    if (apiKey) {
        loadVoices();
    }
});