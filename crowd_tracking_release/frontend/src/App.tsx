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
  const [seconds, setSeconds] = useState(2551);

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

    setAnalytics((prev) => ({
      ...prev,
      total_crowd: total,
      occupancy_rate: parseFloat((total / 32).toFixed(2)),
      density_per_m2: parseFloat((total / 64).toFixed(2)),
      seats_occupied: Math.min(32, total),
      stationary_count: total,
      visual_presentation: {
        female_presenting: female,
        male_presenting: male,
        unknown: unknown,
        coverage_pct: coveragePct,
      },
    }));
  };

  // Session Initialization
  useEffect(() => {
    async function init() {
      const res = await createSession('classroom_demo', currentRoom);
      if (res && res.session) {
        setSessionId(res.session.session_id || res.session.id);
      }
    }
    init();

    const timer = setInterval(() => {
      setSeconds((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [currentRoom]);

  // Periodic Telemetry Poll (~1Hz)
  useEffect(() => {
    if (!sessionId) return;
    const interval = setInterval(async () => {
      const stats = await getSessionStats(sessionId);
      if (stats && stats.analytics) {
        handleAnalyticsUpdate(stats.analytics);
        if (stats.live_stream) {
          setTelemetry((prev) => ({ ...prev, ...stats.live_stream }));
        }
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [sessionId]);

  const formatTime = (totalSeconds: number) => {
    const hrs = Math.floor(totalSeconds / 3600);
    const mins = Math.floor((totalSeconds % 3600) / 60);
    const secs = totalSeconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen flex flex-col font-sans transition-colors duration-300">
      <Header
        currentRoom={currentRoom}
        onRoomChange={setCurrentRoom}
        sessionDuration={formatTime(seconds)}
        lang={lang}
        onToggleLanguage={toggleLanguage}
        t={t}
      />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar activePage={activePage} onPageChange={setActivePage} t={t} />

        <main className="flex-1 overflow-y-auto pb-16 md:pb-0 transition-colors duration-300">
          {activePage === 'overview' && (
            <OverviewPage analytics={analytics} roomCalibrated={true} t={t} />
          )}

          {activePage === 'live' && (
            <LivePage analytics={analytics} onAnalyticsUpdate={handleAnalyticsUpdate} t={t} />
          )}

          {activePage === 'analytics' && <AnalyticsPage analytics={analytics} t={t} />}

          {activePage === 'room' && <RoomSetupPage t={t} />}

          {activePage === 'system' && <SystemPage telemetry={telemetry} t={t} />}

          {activePage === 'video' && <VideoPage />}
        </main>
      </div>

      <Footer telemetry={telemetry} t={t} />
      <MobileNav activePage={activePage} onPageChange={setActivePage} t={t} />
    </div>
  );
}

export default App;
