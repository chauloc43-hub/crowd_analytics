/* ==========================================================================
   CYBER-HUD APPLICATION LOGIC
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Controls & Modes
  const btnModeCam = document.getElementById('btnModeCam');
  const btnModeFile = document.getElementById('btnModeFile');
  const btnSwitchCamera = document.getElementById('btnSwitchCamera');
  const btnStartStream = document.getElementById('btnStartStream');
  const btnStopStream = document.getElementById('btnStopStream');
  const connectionStatus = document.getElementById('connectionStatus');

  // DOM Elements - Video & Canvas
  const webcamVideo = document.getElementById('webcamVideo');
  const outputCanvas = document.getElementById('outputCanvas');
  const canvasCtx = outputCanvas.getContext('2d');
  const fileDropzone = document.getElementById('fileDropzone');
  const fileInput = document.getElementById('fileInput');

  // DOM Elements - Micro Overlay
  const valFps = document.getElementById('valFps');
  const valLatency = document.getElementById('valLatency');

  // DOM Elements - Metrics Cards
  const valTotalCount = document.getElementById('valTotalCount');
  const valMaleCount = document.getElementById('valMaleCount');
  const valFemaleCount = document.getElementById('valFemaleCount');
  const valDensity = document.getElementById('valDensity');
  const barMale = document.getElementById('barMale');
  const barFemale = document.getElementById('barFemale');
  const valGenderRatioPct = document.getElementById('valGenderRatioPct');
  const logConsole = document.getElementById('logConsole');

  // State Variables
  let isStreaming = false;
  let activeMode = 'camera'; // 'camera' or 'file'
  let facingMode = 'user'; // 'user' (Front) or 'environment' (Back)
  let mediaStream = null;
  let frameIntervalTimer = null;
  let sessionID = null;
  let lastFrameTime = performance.now();

  // Logging Utility Function
  function addLog(msg, type = 'normal') {
    const timeStr = new Date().toLocaleTimeString('vi-VN');
    const logItem = document.createElement('div');
    logItem.className = `log-item ${type}`;
    logItem.innerText = `[${timeStr}] ${msg}`;
    logConsole.appendChild(logItem);
    logConsole.scrollTop = logConsole.scrollHeight;

    while (logConsole.children.length > 50) {
      logConsole.removeChild(logConsole.firstChild);
    }
  }

  // Switch Between Camera Stream and File Upload Modes
  btnModeCam.addEventListener('click', () => setMode('camera'));
  btnModeFile.addEventListener('click', () => setMode('file'));

  function setMode(mode) {
    activeMode = mode;
    if (mode === 'camera') {
      btnModeCam.classList.add('active');
      btnModeFile.classList.remove('active');
      webcamVideo.style.display = 'block';
      fileDropzone.style.display = 'none';
      btnSwitchCamera.style.display = 'flex';
      addLog('Đã chuyển sang chế độ Camera Live Feed', 'system');
    } else {
      btnModeFile.classList.add('active');
      btnModeCam.classList.remove('active');
      webcamVideo.style.display = 'none';
      fileDropzone.style.display = 'flex';
      btnSwitchCamera.style.display = 'none';
      stopCameraStream();
      addLog('Đã chuyển sang chế độ Phân tích Video File', 'system');
    }
  }

  // Switch Front / Back Camera (Mobile Facing Mode)
  btnSwitchCamera.addEventListener('click', async () => {
    facingMode = (facingMode === 'user') ? 'environment' : 'user';
    addLog(`Đang chuyển sang Camera (${facingMode === 'user' ? 'Trước' : 'Sau'})...`, 'system');
    if (isStreaming) {
      stopCameraStream();
      await startCameraStream();
    }
  });

  // Start Camera Stream & Create Tracker Session
  btnStartStream.addEventListener('click', () => {
    if (activeMode === 'camera') {
      startCameraStream();
    } else {
      if (fileInput.files.length > 0) {
        uploadAndProcessVideo(fileInput.files[0]);
      } else {
        fileInput.click();
      }
    }
  });

  // Stop Camera Stream
  btnStopStream.addEventListener('click', () => {
    stopCameraStream();
  });

  async function startCameraStream() {
    try {
      addLog('Đang khởi tạo Session theo dõi...', 'system');
      const sessionRes = await fetch('/api/v1/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'default' })
      });

      if (sessionRes.ok) {
        const sessionData = await sessionRes.json();
        sessionID = sessionData.session.id;
        addLog(`Đã khởi tạo Session ID: ${sessionID}`, 'system');
      }

      addLog('Đang yêu cầu quyền truy cập Camera...', 'system');
      const constraints = {
        video: {
          facingMode: facingMode,
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      };

      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      webcamVideo.srcObject = mediaStream;
      await webcamVideo.play();

      isStreaming = true;
      btnStartStream.disabled = true;
      btnStopStream.disabled = false;
      connectionStatus.className = 'hud-status-badge badge-online';
      connectionStatus.querySelector('.status-label').innerText = 'LIVE STREAM';

      addLog('Camera đã kết nối thành công!', 'system');

      startFrameProcessingLoop();
    } catch (err) {
      addLog(`Không thể mở Camera: ${err.message}`, 'error');
      connectionStatus.className = 'hud-status-badge badge-offline';
      connectionStatus.querySelector('.status-label').innerText = 'ERROR';
    }
  }

  function stopCameraStream() {
    if (frameIntervalTimer) {
      clearInterval(frameIntervalTimer);
      frameIntervalTimer = null;
    }

    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }

    if (sessionID) {
      fetch(`/api/v1/sessions/${sessionID}`, { method: 'DELETE' }).catch(() => {});
      addLog(`Đã đóng Session ID: ${sessionID}`, 'system');
      sessionID = null;
    }

    isStreaming = false;
    btnStartStream.disabled = false;
    btnStopStream.disabled = true;
    connectionStatus.className = 'hud-status-badge badge-offline';
    connectionStatus.querySelector('.status-label').innerText = 'STOPPED';

    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    addLog('Đã dừng Stream.', 'warn');
  }

  // Frame Capture & API Transmission Loop
  function startFrameProcessingLoop() {
    if (frameIntervalTimer) clearInterval(frameIntervalTimer);

    frameIntervalTimer = setInterval(() => {
      if (!isStreaming || webcamVideo.paused || webcamVideo.ended) return;

      const now = performance.now();
      const delta = now - lastFrameTime;
      lastFrameTime = now;
      const currentFps = (1000 / delta).toFixed(1);
      valFps.innerText = currentFps;

      outputCanvas.width = webcamVideo.videoWidth || 640;
      outputCanvas.height = webcamVideo.videoHeight || 480;

      canvasCtx.drawImage(webcamVideo, 0, 0, outputCanvas.width, outputCanvas.height);

      outputCanvas.toBlob(async (blob) => {
        if (!blob || !isStreaming) return;

        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        const sendTime = performance.now();
        try {
          const endpoint = sessionID ? `/api/v1/sessions/${sessionID}/frame` : '/api/v1/video/analyze';
          const res = await fetch(endpoint, {
            method: 'POST',
            body: formData
          });

          if (res.ok) {
            const latency = Math.round(performance.now() - sendTime);
            valLatency.innerText = `${latency}ms`;
            const data = await res.json();
            
            // Render annotated frame image from AI backend onto Canvas
            if (data.annotated_frame) {
              const img = new Image();
              img.onload = () => {
                canvasCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
              };
              img.src = data.annotated_frame;
            }

            updateAnalyticsUI(data.analytics);
          }
        } catch (e) {
          // Ignore transient network frame drops
        }
      }, 'image/jpeg', 0.8);

    }, 300);
  }

  // Update UI Stats Cards & Ratio Progress Bar
  function updateAnalyticsUI(stats) {
    if (!stats) return;

    const crowd = stats.crowd || {};
    const genderCounts = crowd.gender_counts || {};

    const total = crowd.current_count ?? 0;
    const male = genderCounts.male ?? 0;
    const female = genderCounts.female ?? 0;

    valTotalCount.innerText = total;
    valMaleCount.innerText = male;
    valFemaleCount.innerText = female;

    // Density Level Calculation based on crowd count
    if (total > 15) {
      valDensity.innerText = 'HIGH';
      valDensity.className = 'metric-value text-neon-pink';
    } else if (total > 5) {
      valDensity.innerText = 'MEDIUM';
      valDensity.className = 'metric-value text-neon-yellow';
    } else {
      valDensity.innerText = 'LOW';
      valDensity.className = 'metric-value text-neon-green';
    }

    // Gender Ratio Calculation
    const totalGender = male + female;
    if (totalGender > 0) {
      const malePct = Math.round((male / totalGender) * 100);
      const femalePct = 100 - malePct;
      barMale.style.width = `${malePct}%`;
      barFemale.style.width = `${femalePct}%`;
      valGenderRatioPct.innerText = `${malePct}% / ${femalePct}%`;
    }
  }

  // Handle Video File Upload
  function uploadAndProcessVideo(file) {
    if (!file) return;
    addLog(`Đang gửi Video File: ${file.name}...`, 'system');

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/v1/video/analyze', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      addLog(`Phân tích Video hoàn tất!`, 'system');
      updateAnalyticsUI(data.analytics);
    })
    .catch(err => {
      addLog(`Lỗi xử lý Video: ${err.message}`, 'error');
    });
  }

  // File Dropzone Event Listeners
  fileDropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadAndProcessVideo(e.target.files[0]);
  });
});
