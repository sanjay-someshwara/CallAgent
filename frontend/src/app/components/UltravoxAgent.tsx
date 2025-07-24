'use client';
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Mic, StopCircle, Disc, Settings } from 'lucide-react';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { toBlobURL, fetchFile } from '@ffmpeg/util';

const UltravoxAgent: React.FC = () => {
  const [ffmpeg, setFFmpeg] = useState<FFmpeg | null>(null);
  const [message, setMessage] = useState<string>('Loading...');
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [aiResponse, setAiResponse] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('openai/whisper-large-v3-turbo');
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(true);
  const [isChangingModel, setIsChangingModel] = useState<boolean>(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserNodeRef = useRef<AnalyserNode | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const silenceTimeoutIdRef = useRef<number | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const hasStartedRecordingRef = useRef<boolean>(false);
  const isCallLiveRef = useRef<boolean>(false);

  const SILENCE_THRESHOLD = 0.08;
  const ANALYSER_FFT_SIZE = 512;

  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:8002/models');
        const data = await response.json();
        setModels(data.models);
        setIsLoadingModels(false);
      } catch (error) {
        console.error('Error fetching models:', error);
        setMessage('Error loading models. Using default model.');
        setIsLoadingModels(false);
      }
    };

    fetchModels();
  }, []);

  // Set model function
  const setModel = async (modelName: string) => {
    setIsChangingModel(true);
    try {
      const response = await fetch('http://localhost:8002/setmodel', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model: modelName }),
      });
      
      if (response.ok) {
        const result = await response.json();
        setSelectedModel(modelName);
        setMessage(result.message || `Model changed to: ${modelName}`);
      } else {
        setMessage('Error changing model. Please try again.');
      }
    } catch (error) {
      console.error('Error setting model:', error);
      setMessage('Error changing model. Please try again.');
    } finally {
      setIsChangingModel(false);
    }
  };

  // Handle model selection change
  const handleModelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = event.target.value;
    setModel(newModel);
  };

  useEffect(() => {
    async function loadFFmpeg() {
      setMessage('Loading FFmpeg...');
      const ffmpegInstance = new FFmpeg();
      // ffmpegInstance.on('log', ({ message }) => setMessage(message));
      const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd';
      await ffmpegInstance.load({
        coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
        wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
      });
      setFFmpeg(ffmpegInstance);
      setMessage('FFmpeg loaded and ready.');
    }
    
    loadFFmpeg();

    return () => {
      ffmpeg?.terminate();
    };
  }, []);

  const stopAudioProcessing = useCallback(async () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamSourceRef.current?.disconnect();
    analyserNodeRef.current?.disconnect();
    audioContextRef.current?.close();

    mediaStreamSourceRef.current = null;
    analyserNodeRef.current = null;
    audioContextRef.current = null;

    if (silenceTimeoutIdRef.current) clearTimeout(silenceTimeoutIdRef.current);
    if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);

    silenceTimeoutIdRef.current = null;
    animationFrameIdRef.current = null;
    hasStartedRecordingRef.current = false;

    isCallLiveRef.current = false;
    setIsSpeaking(false);
    setIsRecording(false);
    setMessage('Processing audio...');

    // Send the recorded audio to backend
    if (audioChunksRef.current.length > 0) {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      await convertToWav(audioBlob);
      audioChunksRef.current = [];
    }
    
    setMessage('Recording stopped. Audio sent to backend.');
  }, []);

  const detectVoiceActivity = useCallback(() => {
    if (!analyserNodeRef.current || !audioContextRef.current) return;
    const analyser = analyserNodeRef.current;
    const data = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(data);
    const rms = Math.sqrt(data.reduce((sum, val) => sum + val * val, 0) / data.length);

    if (rms > SILENCE_THRESHOLD) {
      setIsSpeaking(true);
    } else {
      setIsSpeaking(false);
    }

    animationFrameIdRef.current = requestAnimationFrame(detectVoiceActivity);
  }, []);

  const startCall = async () => {
    if (!ffmpeg) {
      setMessage('FFmpeg is not loaded yet.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      audioChunksRef.current = [];

      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      mediaRecorderRef.current.onstop = async () => {
        // Only process if we're stopping the recording (not just a chunk)
        if (!isCallLiveRef.current && audioChunksRef.current.length > 0) {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          await convertToWav(audioBlob);
        }
      };

      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      mediaStreamSourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      analyserNodeRef.current = audioContextRef.current.createAnalyser();
      analyserNodeRef.current.fftSize = ANALYSER_FFT_SIZE;
      mediaStreamSourceRef.current.connect(analyserNodeRef.current);

      isCallLiveRef.current = true;
      setIsRecording(true);
      setMessage('Recording... Click "Stop Recording" when done.');
      mediaRecorderRef.current.start();
      animationFrameIdRef.current = requestAnimationFrame(detectVoiceActivity);
    } catch (error) {
      console.error('Error:', error);
      setMessage('Microphone error.');
      stopAudioProcessing();
    }
  };

  const convertToWav = async (inputBlob: Blob) => {
    if (!ffmpeg || !inputBlob) return;
    try {
      const inputFileName = 'input.webm';
      const outputFileName = 'output.wav';
      await ffmpeg.writeFile(inputFileName, await fetchFile(inputBlob));
      await ffmpeg.exec(['-i', inputFileName, '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', outputFileName]);
      const data = await ffmpeg.readFile(outputFileName);
      const wavBlob = new Blob([data], { type: 'audio/wav' });

      const formData = new FormData();
      formData.append('file', wavBlob, 'recorded_audio.wav');
      // Use /inference endpoint and include sessionId as query param
      const url = `http://localhost:8002/inference${sessionId ? `?session_id=${sessionId}` : ''}`;
      const response = await fetch(url, { method: 'POST', body: formData });
      const result = await response.json();
      if (result.error) {
        setMessage(`Error: ${result.error}`);
        setTranscription(null);
        setAiResponse(null);
      } else {
        setSessionId(result.session_id);
        setTranscription(result.transcription);
        setAiResponse(result.ultravox_response);
        setMessage('Response received from Ultravox!');
      }
      await ffmpeg.deleteFile(inputFileName);
      await ffmpeg.deleteFile(outputFileName);
    } catch (error) {
      console.error('Conversion/Upload Error:', error);
      setMessage('Error processing audio. Please try again.');
      setTranscription(null);
      setAiResponse(null);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4 font-sans rounded-lg">
      <h1 className="text-4xl font-extrabold text-blue-800 mb-8 drop-shadow-md">Ultravox Agent</h1>
      
      {/* Model Selection */}
      <div className="mb-6 p-4 bg-white rounded-lg shadow-md w-full max-w-md">
        <div className="flex items-center mb-3">
          <Settings className="h-5 w-5 text-gray-600 mr-2" />
          <label htmlFor="model-select" className="text-sm font-medium text-gray-700">
            Whisper Model
          </label>
        </div>
        <select
          id="model-select"
          value={selectedModel}
          onChange={handleModelChange}
          disabled={isLoadingModels || isChangingModel}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 text-black"
        >
          {isLoadingModels ? (
            <option className="text-black">Loading models...</option>
          ) : (
            models.map((model) => (
              <option key={model} value={model} className="text-black">
                {model.replace('openai/whisper-', '')}
              </option>
            ))
          )}
        </select>
        {isChangingModel && (
          <div className="flex items-center justify-center mt-2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            <span className="ml-2 text-sm text-gray-600">Changing model...</span>
          </div>
        )}
        {!isLoadingModels && (
          <p className="text-xs text-gray-500 mt-1">
            Current: {selectedModel.replace('openai/whisper-', '')}
          </p>
        )}
      </div>

      <p className="text-gray-700 text-lg mb-6 text-center">{message}</p>
      {transcription && (
        <div className="mb-4 p-4 bg-white rounded shadow w-full max-w-xl">
          <strong>Transcription:</strong> {transcription}
        </div>
      )}
      {aiResponse && (
        <div className="mb-4 p-4 bg-white rounded shadow w-full max-w-xl text-black">
          <strong>Ultravox Response:</strong> {aiResponse}
        </div>
      )}
      
      <div className="flex space-x-6 mb-8">
        <button onClick={startCall} disabled={isRecording || !ffmpeg} className="px-8 py-4 text-lg font-semibold rounded-full bg-green-600 hover:bg-green-700 text-white disabled:opacity-50 disabled:cursor-not-allowed">
          <Mic className="mr-3 h-6 w-6" /> Start Recording
        </button>
        <button onClick={stopAudioProcessing} disabled={!isRecording} className="px-8 py-4 text-lg font-semibold rounded-full bg-red-600 hover:bg-red-700 text-white disabled:opacity-50 disabled:cursor-not-allowed">
          <StopCircle className="mr-3 h-6 w-6" /> Stop Recording
        </button>
      </div>
      <div className="relative w-48 h-48 flex items-center justify-center mb-8">
        <div className={`absolute inset-0 bg-blue-500 rounded-full transition-all duration-300 ${isSpeaking && isRecording ? 'scale-105 opacity-80 animate-pulse-light' : 'scale-75 opacity-20'}`} />
        <Disc className={`h-24 w-24 text-blue-700 transition-transform duration-300 ${isSpeaking && isRecording ? 'scale-110' : 'scale-90'}`} />
      </div>
    </div>
  );
};

export default UltravoxAgent;