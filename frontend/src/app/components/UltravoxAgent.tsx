'use client';
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Mic, StopCircle, Disc } from 'lucide-react';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { toBlobURL, fetchFile } from '@ffmpeg/util';

const UltravoxAgent: React.FC = () => {
  const [ffmpeg, setFFmpeg] = useState<FFmpeg | null>(null);
  const [message, setMessage] = useState<string>('Loading...');
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);

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
  const SILENCE_DURATION = 1500;
  const ANALYSER_FFT_SIZE = 512;

  useEffect(() => {
    async function loadFFmpeg() {
      setMessage('Loading FFmpeg...');
      const ffmpegInstance = new FFmpeg();
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
      const response = await fetch('http://localhost:8000/transcribe', { method: 'POST', body: formData });

      const transcription = await response.json();
      console.log('Transcription:', transcription);
      
      if (transcription.error) {
        setMessage(`Error: ${transcription.error}`);
      } else {
        setMessage(`Transcription (${transcription.model_used}): ${transcription.transcription}`);
      }

      await ffmpeg.deleteFile(inputFileName);
      await ffmpeg.deleteFile(outputFileName);
    } catch (error) {
      console.error('Conversion/Upload Error:', error);
      setMessage('Error processing audio. Please try again.');
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4 font-sans rounded-lg">
      <h1 className="text-4xl font-extrabold text-blue-800 mb-8 drop-shadow-md">Ultravox Agent</h1>
      <p className="text-gray-700 text-lg mb-6 text-center">{message}</p>
      
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