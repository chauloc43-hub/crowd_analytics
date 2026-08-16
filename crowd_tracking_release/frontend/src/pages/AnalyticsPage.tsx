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
                    <span className="text-cyan-400 font-bold">23% (6 people)</span>
                  </div>
                  <div className="w-full h-3 bg-[#071120] rounded-full overflow-hidden border border-sky-500/30">
                    <div className="h-full bg-cyan-400 rounded-full transition-all duration-300" style={{ width: '23%' }} />
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
                    <span className="text-cyan-400 font-bold">54% (14 people)</span>
                  </div>
                  <div className="w-full h-3 bg-[#071120] rounded-full overflow-hidden border border-sky-500/30">
                    <div className="h-full bg-sky-400 rounded-full transition-all duration-300" style={{ width: '54%' }} />
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
                    <span className="text-cyan-400 font-bold">23% (6 people)</span>
                  </div>
                  <div className="w-full h-3 bg-[#071120] rounded-full overflow-hidden border border-sky-500/30">
                    <div className="h-full bg-purple-500 rounded-full transition-all duration-300" style={{ width: '23%' }} />
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
                    <td className="px-4 py-3 font-bold">6</td>
                    <td className="px-4 py-3">0.38 /m²</td>
                    <td className="px-4 py-3 text-slate-300">14m 02s</td>
                  </tr>
                  <tr className="hover:bg-sky-500/10 transition-colors">
                    <td className="px-4 py-3 font-bold text-cyan-400">Middle</td>
                    <td className="px-4 py-3 font-bold">14</td>
                    <td className="px-4 py-3 font-bold text-cyan-300">0.50 /m²</td>
                    <td className="px-4 py-3 text-slate-300">31m 15s</td>
                  </tr>
                  <tr className="hover:bg-sky-500/10 transition-colors">
                    <td className="px-4 py-3 font-bold text-cyan-400">Back</td>
                    <td className="px-4 py-3 font-bold">6</td>
                    <td className="px-4 py-3">0.35 /m²</td>
                    <td className="px-4 py-3 text-slate-300">27m 41s</td>
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
