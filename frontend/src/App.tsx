import { useState, useEffect } from 'react';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { MobileNav } from './components/layout/MobileNav';
import { Footer } from './components/layout/Footer';
import { OverviewPage } from './pages/OverviewPage';
import { LivePage } from './pages/LivePage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { RoomSetupPage } from './pages/RoomSetupPage';
import { SystemPage } from './pages/SystemPage';
import { VideoPage } from './pages/VideoPage';
import type { PageType, AnalyticsData, LiveStreamTelemetry } from './types/analytics';
import { createSession, getSessionStats } from './api/crowdApi';
import { translations, type Language } from './i18n/translations';

export function App() {
  const [activePage, setActivePage] = useState<PageType>('overview');
  const [currentRoom, setCurrentRoom] = useState('Classroom A');
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Language State ('vi' | 'en') - Default to Vietnamese ('vi')
  const [lang, setLang] = useState<Language>(() => {
    return (localStorage.getItem('crowd_analytics_lang') as Language) || 'vi';
  });

  const toggleLanguage = () => {
    const nextLang = lang === 'vi' ? 'en' : 'vi';
    setLang(nextLang);
    localStorage.setItem('crowd_analytics_lang', nextLang);
  };

  const t = translations[lang];

  // Analytics data state
  const [analytics, setAnalytics] = useState<AnalyticsData>({
    total_crowd: 0,
    occupancy_rate: 0.0,
    density_per_m2: 0.0,
    seats_occupied: 0,
    total_seats: 32,
    moving_count: 0,
    stationary_count: 0,
    flow_in: 0,
    flow_out: 0,
    net_flow: 0,
    visual_presentation: {
      female_presenting: 0,
      male_presenting: 0,
      unknown: 0,
      coverage_pct: 100,
    },
    space_distribution: {
      front_pct: 0,
      middle_pct: 0,
      back_pct: 0,
    },
    zones: [
      { name: 'Front', peopleCount: 0, density: 0.0, avgDwellTime: '0m 00s', percentage: 0 },
      { name: 'Middle', peopleCount: 0, density: 0.0, avgDwellTime: '0m 00s', percentage: 0 },
      { name: 'Back', peopleCount: 0, density: 0.0, avgDwellTime: '0m 00s', percentage: 0 },
    ],
  });

  const [telemetry, setTelemetry] = useState<LiveStreamTelemetry>({
    received_frames: 0,
    processed_frames: 0,
    replaced_frames: 0,
    pending_frames: 0,
    current_fps: 30.0,
    ai_update_rate_hz: 6.7,
    processing_fps: 14.1,
    latency_p50_ms: 68,
    latency_p95_ms: 86,
  });

  // Handle Real AI Model Output from Live Page
  const handleAnalyticsUpdate = (rawStats: any) => {
    if (!rawStats) return;

    const crowd = rawStats.crowd || {};
    const genderCounts = crowd.gender_counts || rawStats.attributes?.gender_counts || {};

    const total = crowd.current_count ?? analytics.total_crowd;
    const female = genderCounts.female ?? 0;
    const male = genderCounts.male ?? 0;
    const unknown = genderCounts.unknown ?? Math.max(0, total - (female + male));
    const totalGender = female + male + unknown;
    const coveragePct = totalGender > 0 ? Math.round(((female + male) / totalGender) * 100) : 100;

    const crossing = rawStats.crossing || {};
    const flowIn = (crossing.in || 0) > 0 ? crossing.in : (crowd.unique_track_count ?? (crowd.current_count || 0));
    const currentCount = crowd.current_count ?? total;
    const flowOut = (crossing.out || 0) > 0 ? crossing.out : Math.max(crowd.finalized_count || 0, Math.max(0, flowIn - currentCount));
    const netFlow = Math.abs(flowIn) + Math.abs(flowOut);

    setAnalytics((prev) => ({
      ...prev,
      total_crowd: total,
      occupancy_rate: parseFloat((total / 32).toFixed(2)),
      density_per_m2: parseFloat((total / 64).toFixed(2)),
      seats_occupied: Math.min(32, total),
      stationary_count: total,
      flow_in: flowIn,
      flow_out: flowOut,
      net_flow: netFlow,
      visual_presentation: {
        female_presenting: female,
        male_presenting: male,
        unknown: unknown,
        coverage_pct: coveragePct,
      },
      spatial: rawStats.spatial || prev.spatial,
    }));
  };

  const [vietnamTime, setVietnamTime] = useState('');
  const [isLiveStreamActive, setIsLiveStreamActive] = useState(false);

  // Live System Event Logs
  const [systemLogs, setSystemLogs] = useState<string[]>(() => [
    `[${new Date().toLocaleTimeString('vi-VN')}] [SYSTEM] Crowd Analytics Telemetry Engine online.`,
    `[${new Date().toLocaleTimeString('vi-VN')}] [INFO] YOLO11n + FastTracker backend pipeline initialized.`,
  ]);

  const addSystemLog = (msg: string) => {
    const timeStr = new Date().toLocaleTimeString('vi-VN');
    setSystemLogs((prev) => [...prev.slice(-99), `[${timeStr}] ${msg}`]);
  };

  // Real-time Vietnam Time (Asia/Ho_Chi_Minh) Clock
  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      const timeStr = now.toLocaleTimeString('vi-VN', {
        timeZone: 'Asia/Ho_Chi_Minh',
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
      setVietnamTime(timeStr);
    };
    updateClock();
    const timer = setInterval(updateClock, 1000);
    return () => clearInterval(timer);
  }, []);

  // Session Initialization
  useEffect(() => {
    async function init() {
      const res = await createSession('classroom_demo', currentRoom);
      if (res && res.session) {
        const sId = res.session.session_id || res.session.id;
        setSessionId(sId);
        addSystemLog(`[INFO] Session initialized: ${sId} (${currentRoom})`);
      }
    }
    init();
  }, [currentRoom]);

  // Periodic Telemetry Poll (~1Hz)
  useEffect(() => {
    if (!sessionId) return;
    const interval = setInterval(async () => {
      const stats = await getSessionStats(sessionId);
      if (stats && stats._session_expired) {
        setSessionId(null);
        addSystemLog(`[WARN] Session ${sessionId} expired or closed.`);
        return;
      }
      if (stats && stats.analytics) {
        handleAnalyticsUpdate(stats.analytics);
        if (stats.live_stream) {
          setTelemetry((prev) => ({ ...prev, ...stats.live_stream }));
        }
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [sessionId]);

  return (
    <div className="min-h-screen flex flex-col font-sans transition-colors duration-300">
      <Header
        currentRoom={currentRoom}
        onRoomChange={setCurrentRoom}
        sessionDuration={vietnamTime}
        lang={lang}
        onToggleLanguage={toggleLanguage}
        t={t}
        isLive={isLiveStreamActive}
      />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar activePage={activePage} onPageChange={setActivePage} t={t} />

        <main className="flex-1 overflow-y-auto pb-16 md:pb-0 transition-colors duration-300">
          {activePage === 'overview' && (
            <OverviewPage analytics={analytics} roomCalibrated={true} t={t} />
          )}

          <div className={activePage === 'live' ? 'block' : 'hidden'}>
            <LivePage
              analytics={analytics}
              onAnalyticsUpdate={handleAnalyticsUpdate}
              t={t}
              onStreamingChange={setIsLiveStreamActive}
              addSystemLog={addSystemLog}
            />
          </div>

          {activePage === 'analytics' && <AnalyticsPage analytics={analytics} t={t} />}

          {activePage === 'room' && <RoomSetupPage t={t} />}

          {activePage === 'system' && (
            <SystemPage telemetry={telemetry} t={t} logs={systemLogs} isLive={isLiveStreamActive} />
          )}

          {activePage === 'video' && <VideoPage />}
        </main>
      </div>

      <Footer telemetry={telemetry} t={t} />
      <MobileNav activePage={activePage} onPageChange={setActivePage} t={t} />
    </div>
  );
}

export default App;
