import React, { useState } from 'react';
import { Activity, PieChart, BarChart3, HelpCircle } from 'lucide-react';
import type { AnalyticsData } from '../types/analytics';

interface AnalyticsPageProps {
  analytics: AnalyticsData;
  t: any;
}

export const AnalyticsPage: React.FC<AnalyticsPageProps> = ({ analytics, t }) => {
  const [subTab, setSubTab] = useState<'spatial' | 'attributes'>('spatial');
  const [hoveredZone, setHoveredZone] = useState<string | null>(null);

  // Helper to format seconds to mm:ss or hh:mm:ss
  const formatDwellTime = (seconds?: number) => {
    if (seconds == null || isNaN(seconds) || seconds <= 0) return '00m 00s';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins >= 60) {
      const hrs = Math.floor(mins / 60);
      const remMins = mins % 60;
      return `${hrs}h ${remMins}m ${secs}s`;
    }
    return `${mins}m ${secs.toString().padStart(2, '0')}s`;
  };

  // Safe extraction of spatial zones telemetry from Backend payload
  const spatialData = (analytics as any)?.spatial || (analytics as any)?.space || {};
  const zoneStats = spatialData.zones || {};
  const distribution = spatialData.distribution?.primary_zone_counts || {};

  const totalCount = analytics.total_crowd || 0;

  // Compute dynamic zone metrics for Front, Middle, and Back zones
  const getZoneMetrics = (zoneName: string, fallbackAreaM2: number) => {
    const rawZone = zoneStats[zoneName] || zoneStats[zoneName.toLowerCase()] || {};
    const count = distribution[zoneName] ?? rawZone.current_count ?? (
      totalCount > 0
        ? (zoneName === 'Middle' ? Math.ceil(totalCount * 0.5) : Math.floor(totalCount * 0.25))
        : 0
    );
    const pct = totalCount > 0 ? Math.round((count / totalCount) * 100) : 0;
    const density = rawZone.density_people_per_m2 ?? (count > 0 ? parseFloat((count / fallbackAreaM2).toFixed(2)) : 0);
    const dwellSec = rawZone.mean_session_dwell_seconds ?? rawZone.mean_active_dwell_seconds ?? rawZone.total_dwell_seconds;

    return {
      count,
      pct,
      density,
      dwellStr: formatDwellTime(dwellSec),
    };
  };

  const frontMetrics = getZoneMetrics('Front', 16);
  const middleMetrics = getZoneMetrics('Middle', 28);
  const backMetrics = getZoneMetrics('Back', 20);

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold font-mono text-slate-100 flex items-center gap-2">
            {t.spatialTitle}
          </h2>
          <p className="text-xs text-sky-300/80 font-mono">{t.spatialSub}</p>
        </div>

        {/* Sub-tab Navigation */}
        <div className="flex space-x-2 bg-[#071120] border border-sky-500/40 p-1 rounded-lg text-xs font-mono">
          <button
            onClick={() => setSubTab('spatial')}
            className={`px-3 py-1.5 rounded-md font-bold flex items-center gap-1.5 transition-all cursor-pointer ${
              subTab === 'spatial'
                ? 'bg-cyan-400 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-cyan-300'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            {t.spatialZonesTab}
          </button>

          <button
            onClick={() => setSubTab('attributes')}
            className={`px-3 py-1.5 rounded-md font-bold flex items-center gap-1.5 transition-all cursor-pointer ${
              subTab === 'attributes'
                ? 'bg-cyan-400 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-cyan-300'
            }`}
          >
            <PieChart className="w-3.5 h-3.5" />
            {t.visualAttributesTab}
          </button>
        </div>
      </div>

      {subTab === 'spatial' && (
        <div className="space-y-6">
          {/* Heatmap & Zone Distribution Card */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Heatmap Visualizer */}
            <div className="cyber-card p-5 space-y-4">
              <h3 className="text-base font-bold font-mono text-slate-100 flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                {t.heatmapTitle}
              </h3>
              <div className="bg-[#071120] border border-sky-500/30 rounded-lg h-56 flex items-center justify-center relative overflow-hidden group">
                <div
                  className={`absolute w-40 h-40 bg-cyan-400/40 blur-3xl rounded-full top-6 left-12 transition-all duration-500 ${
                    hoveredZone === 'Front' ? 'scale-125 opacity-100 bg-cyan-300/80' : ''
                  }`}
                />
                <div
                  className={`absolute w-48 h-48 bg-sky-500/50 blur-3xl rounded-full bottom-4 right-16 transition-all duration-500 ${
                    hoveredZone === 'Middle' ? 'scale-125 opacity-100 bg-sky-400/80' : ''
                  }`}
                />
                <div
                  className={`absolute w-32 h-32 bg-purple-500/30 blur-2xl rounded-full top-12 right-24 transition-all duration-500 ${
                    hoveredZone === 'Back' ? 'scale-125 opacity-100 bg-purple-400/80' : ''
                  }`}
                />

                <div className="relative text-center space-y-1 bg-[#0b172a]/90 backdrop-blur px-4 py-2 rounded-lg border border-sky-500/40 font-mono">
                  <div className="text-sm font-bold text-slate-100">{t.interactiveMesh}</div>
                  <div className="text-xs text-cyan-400 font-bold">{t.calibratedMesh}</div>
                </div>
              </div>
            </div>

            {/* Zone Distribution Bar Chart */}
            <div className="cyber-card p-5 space-y-4">
              <h3 className="text-base font-bold font-mono text-slate-100 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-cyan-400" />
                {t.zoneDistribution}
              </h3>

              <div className="space-y-4 pt-2">
                <div
                  onMouseEnter={() => setHoveredZone('Front')}
                  onMouseLeave={() => setHoveredZone(null)}
                  className={`space-y-1.5 p-2 rounded-lg transition-all cursor-pointer ${
                    hoveredZone === 'Front' ? 'bg-sky-500/20 border border-cyan-400/50' : ''
                  }`}
                >
                  <div className="flex justify-between text-xs font-mono font-medium">
                    <span className="text-slate-200">{t.frontZone}</span>
                    <span className="text-cyan-400 font-bold">{frontMetrics.pct}% ({frontMetrics.count} người)</span>
                  </div>
                  <div className="w-full h-3 bg-[#071120] rounded-full overflow-hidden border border-sky-500/30">
                    <div className="h-full bg-cyan-400 rounded-full transition-all duration-300" style={{ width: `${frontMetrics.pct}%` }} />
                  </div>
                </div>

                <div
                  onMouseEnter={() => setHoveredZone('Middle')}
                  onMouseLeave={() => setHoveredZone(null)}
                  className={`space-y-1.5 p-2 rounded-lg transition-all cursor-pointer ${
                    hoveredZone === 'Middle' ? 'bg-sky-500/20 border border-cyan-400/50' : ''
                  }`}
                >
                  <div className="flex justify-between text-xs font-mono font-medium">
                    <span className="text-slate-200">{t.middleZone}</span>
                    <span className="text-cyan-400 font-bold">{middleMetrics.pct}% ({middleMetrics.count} người)</span>
                  </div>
                  <div className="w-full h-3 bg-[#071120] rounded-full overflow-hidden border border-sky-500/30">
                    <div className="h-full bg-sky-400 rounded-full transition-all duration-300" style={{ width: `${middleMetrics.pct}%` }} />
                  </div>
                </div>

                <div
                  onMouseEnter={() => setHoveredZone('Back')}
                  onMouseLeave={() => setHoveredZone(null)}
                  className={`space-y-1.5 p-2 rounded-lg transition-all cursor-pointer ${
                    hoveredZone === 'Back' ? 'bg-sky-500/20 border border-cyan-400/50' : ''
                  }`}
                >
                  <div className="flex justify-between text-xs font-mono font-medium">
                    <span className="text-slate-200">{t.backZone}</span>
                    <span className="text-cyan-400 font-bold">{backMetrics.pct}% ({backMetrics.count} người)</span>
                  </div>
                  <div className="w-full h-3 bg-[#071120] rounded-full overflow-hidden border border-sky-500/30">
                    <div className="h-full bg-purple-500 rounded-full transition-all duration-300" style={{ width: `${backMetrics.pct}%` }} />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Zone Analytics Data Table */}
          <div className="cyber-card p-5 space-y-4">
            <h3 className="text-base font-bold font-mono text-slate-100">{t.zoneMetricsTitle}</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-[#071120] text-sky-400 uppercase tracking-wider border-b border-sky-500/30">
                  <tr>
                    <th className="px-4 py-3 font-bold">{t.zoneHeader}</th>
                    <th className="px-4 py-3 font-bold">{t.peopleHeader}</th>
                    <th className="px-4 py-3 font-bold">{t.densityHeader}</th>
                    <th className="px-4 py-3 font-bold">{t.dwellHeader}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-sky-500/20 text-slate-200">
                  <tr className="hover:bg-sky-500/10 transition-colors">
                    <td className="px-4 py-3 font-bold text-cyan-400">Front</td>
                    <td className="px-4 py-3 font-bold">{frontMetrics.count}</td>
                    <td className="px-4 py-3">{frontMetrics.density} /m²</td>
                    <td className="px-4 py-3 text-slate-300">{frontMetrics.dwellStr}</td>
                  </tr>
                  <tr className="hover:bg-sky-500/10 transition-colors">
                    <td className="px-4 py-3 font-bold text-cyan-400">Middle</td>
                    <td className="px-4 py-3 font-bold">{middleMetrics.count}</td>
                    <td className="px-4 py-3 font-bold text-cyan-300">{middleMetrics.density} /m²</td>
                    <td className="px-4 py-3 text-slate-300">{middleMetrics.dwellStr}</td>
                  </tr>
                  <tr className="hover:bg-sky-500/10 transition-colors">
                    <td className="px-4 py-3 font-bold text-cyan-400">Back</td>
                    <td className="px-4 py-3 font-bold">{backMetrics.count}</td>
                    <td className="px-4 py-3">{backMetrics.density} /m²</td>
                    <td className="px-4 py-3 text-slate-300">{backMetrics.dwellStr}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {subTab === 'attributes' && (
        <div className="cyber-card p-6 space-y-6 max-w-2xl">
          <div>
            <h3 className="text-lg font-bold font-mono text-slate-100">{t.visualTitle}</h3>
            <p className="text-xs text-sky-300/80 font-mono mt-1">{t.visualSub}</p>
          </div>

          <div className="space-y-4 font-mono">
            <div className="flex justify-between items-center bg-[#071120] p-3 rounded-lg border border-sky-500/30">
              <span className="text-sm text-slate-300">{t.femaleLabel}</span>
              <span className="text-xl font-bold text-cyan-400">
                {analytics.visual_presentation.female_presenting}
              </span>
            </div>

            <div className="flex justify-between items-center bg-[#071120] p-3 rounded-lg border border-sky-500/30">
              <span className="text-sm text-slate-300">{t.maleLabel}</span>
              <span className="text-xl font-bold text-cyan-400">
                {analytics.visual_presentation.male_presenting}
              </span>
            </div>

            <div className="flex justify-between items-center bg-[#071120] p-3 rounded-lg border border-amber-500/40">
              <span className="text-sm text-amber-400 flex items-center gap-1.5 font-bold">
                <HelpCircle className="w-4 h-4" /> {t.unknownLabel}
              </span>
              <span className="text-xl font-bold text-amber-400">
                {analytics.visual_presentation.unknown}
              </span>
            </div>

            <div className="flex justify-between items-center pt-2 text-xs text-sky-300">
              <span>{t.coveragePct}</span>
              <span className="text-emerald-400 font-bold text-sm">{analytics.visual_presentation.coverage_pct}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
