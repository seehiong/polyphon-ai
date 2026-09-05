// Polyphon Studio Global State & Constants

// Studio Session & Data
let currentUploadedFile = null;
let activeResultData = null;
let currentLoadedMeetingId = null;
let currentMeetingPayload = null;
let currentTranscriptViewMode = 'processed'; // 'processed' | 'live'
let activeSpeakerFilter = 'all';
const SPEAKER_COLORS = [
  '#3b82f6', '#10b981', '#8b5cf6', '#f59e0b',
  '#ec4899', '#06b6d4', '#14b8a6', '#f97316'
];

// Streaming State
let streamSocket = null;
let audioContext = null;
let micStream = null;
let displayStream = null; // Phase 6.1: Meeting tab/system audio stream
let processorNode = null;
let isLiveStreaming = false;
let streamTimerInterval = null;
let simInterval = null;
let streamStartTime = 0;
let currentLiveSessionName = null;
let lastRecordedLiveSession = null;
let streamSource = 'mic'; // 'mic' | 'dual' | 'sim'
let customSimFile = null;

// Processing & SSE Job State
let currentActiveJob = null;
let activeEventSource = null;

// Media Player & Karaoke State
let activePlayer = null;
let currentActiveCard = null;
let turnCardIndex = [];
let isVideoMode = false;
let isAutoScrollEnabled = true;
let isCleanVerbatimEnabled = false;
let currentPlaybackRate = 1.0;

// Enrollment State
let currentEnrollSpeakerId = null;

