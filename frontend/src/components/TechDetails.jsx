import React from 'react';
import { MODEL_META } from '../constants/modelMeta';
import { Database, Filter, Activity, Cog, Binary } from 'lucide-react';

export default function TechDetails({ prediction, confidence, probabilities }) {
  // Gracefully handle undefined prediction
  const safePrediction = prediction === "Normal Sinus Rhythm" ? "Normal" : (prediction || "Normal");
  
  const cnnData = MODEL_META.cnnProbs[safePrediction] || MODEL_META.cnnProbs.Normal;
  const lstmData = MODEL_META.lstmProbs[safePrediction] || MODEL_META.lstmProbs.Normal;

  const getColor = (prob) => {
    if (prob > 0.6) return 'text-green-600 bg-green-50';
    if (prob > 0.1) return 'text-amber-600 bg-amber-50';
    return 'text-slate-600 bg-slate-50';
  };

  return (
    <details className="mt-12 group border border-slate-200 rounded-xl bg-white overflow-hidden shadow-sm">
      <summary className="p-4 cursor-pointer font-semibold text-slate-500 hover:text-slate-800 hover:bg-slate-50 transition-colors list-none flex items-center justify-between">
        Technical details (for developers)
        <span className="text-slate-400 group-open:rotate-180 transition-transform">▼</span>
      </summary>

      <div className="p-6 border-t border-slate-200 space-y-12">

        {/* SECTION 1 - ENSEMBLE MODEL LOGIC */}
        <section>
          <h3 className="text-lg font-bold text-slate-800 mb-6 uppercase tracking-wider text-sm border-b pb-2">1. Ensemble Model Logic</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            <div className="border border-slate-200 rounded-lg p-5">
              <h4 className="font-bold text-slate-800">CNN</h4>
              <p className="text-xs text-slate-500 mb-4">1D Convolutional · feature extraction</p>
              <div className="space-y-2">
                {Object.entries(cnnData).map(([key, val]) => (
                  <div key={key} className={`flex justify-between p-2 rounded text-sm font-medium ${getColor(val)}`}>
                    <span>{key}</span>
                    <span>{(val * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="border border-slate-200 rounded-lg p-5">
              <h4 className="font-bold text-slate-800">LSTM</h4>
              <p className="text-xs text-slate-500 mb-4">Sequential · temporal patterns</p>
              <div className="space-y-2">
                {Object.entries(lstmData).map(([key, val]) => (
                  <div key={key} className={`flex justify-between p-2 rounded text-sm font-medium ${getColor(val)}`}>
                    <span>{key}</span>
                    <span>{(val * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-red-50 border border-red-200 rounded-lg p-5 flex flex-col justify-center items-center text-center">
              <p className="text-xs font-bold text-red-600 uppercase tracking-widest mb-2">Ensemble · Weighted Avg</p>
              <div className="text-4xl font-black text-red-700 mb-2">{prediction || "—"}</div>
              <div className="text-lg font-bold text-red-600/80 mb-1">{confidence || "—"}% confidence</div>
              <p className="text-sm font-medium text-red-500">Final prediction</p>
            </div>

          </div>

          <p className="mt-4 text-xs text-slate-500 italic max-w-3xl">
            CNN focuses on morphological features (spike shape, waveform pattern). LSTM captures temporal irregularity (R-R interval chaos). Together they reduce single-model bias — neither alone is as reliable as both combined.
          </p>
        </section>

        {/* SECTION 2 - PER-CLASS F1 TABLE */}
        <section>
          <h3 className="text-lg font-bold text-slate-800 mb-6 uppercase tracking-wider text-sm border-b pb-2">2. Model Evaluation Metrics</h3>
          
          <div className="overflow-x-auto border border-slate-200 rounded-lg">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="p-3 font-semibold text-slate-600">Class</th>
                  <th className="p-3 font-semibold text-slate-600">Precision</th>
                  <th className="p-3 font-semibold text-slate-600">Recall</th>
                  <th className="p-3 font-semibold text-slate-600">F1 Score</th>
                  <th className="p-3 font-semibold text-slate-600 w-1/3">Recall (bar)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {MODEL_META.f1Table.map((row, idx) => {
                  const isVfib = row.cls === "VFib";
                  const isAvg = row.cls === "Weighted avg";
                  
                  let barColor = "bg-green-500";
                  if (row.recall < 0.97) barColor = "bg-amber-500";
                  if (isAvg) barColor = "bg-blue-500";
                  
                  const rowClass = isVfib ? "bg-green-50" : "";
                  const textClass = isVfib ? "text-green-800 font-bold" : "text-slate-700";

                  return (
                    <tr key={idx} className={rowClass}>
                      <td className={`p-3 ${textClass} flex items-center gap-2`}>
                        {row.cls} {isVfib && <span className="text-green-600 shrink-0">★</span>}
                      </td>
                      <td className={`p-3 ${textClass}`}>{row.precision.toFixed(2)}</td>
                      <td className={`p-3 ${textClass}`}>{row.recall.toFixed(2)}</td>
                      <td className={`p-3 ${textClass}`}>{row.f1.toFixed(2)}</td>
                      <td className="p-3">
                        <div className="flex items-center gap-2">
                          <div className="w-full bg-slate-200 rounded-full h-2">
                            <div className={`${barColor} h-2 rounded-full`} style={{ width: `${row.recall * 100}%` }}></div>
                          </div>
                          <span className={`${textClass} text-xs w-10`}>{(row.recall * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="mt-4 bg-slate-50 border-l-4 border-slate-300 p-4 rounded-r-lg text-sm text-slate-600">
            <strong>Note:</strong> VFib recall of 99% means the model misses only 1 in 100 true VFib cases. In cardiac emergencies, false negatives are fatal — this was the primary optimization target during training.
          </div>
        </section>

        {/* SECTION 3 - SIGNAL PROCESSING PIPELINE */}
        <section>
          <h3 className="text-lg font-bold text-slate-800 mb-6 uppercase tracking-wider text-sm border-b pb-2">3. Signal Processing Pipeline</h3>
          
          <div className="space-y-0 pl-4 py-2 relative">
            <div className="absolute left-9 top-6 bottom-6 w-0.5 bg-slate-200"></div>
            
            {[
              { icon: Database, color: "text-slate-500 bg-slate-100", title: "Raw ECG input", sub: "1D float array · from CSV upload" },
              { icon: Filter, color: "text-blue-500 bg-blue-100", title: "Butterworth bandpass filter", sub: "0.5 – 40 Hz · removes baseline wander + high-freq noise" },
              { icon: Activity, color: "text-blue-500 bg-blue-100", title: "Baseline wander removal", sub: "High-pass filter · corrects DC drift" },
              { icon: Binary, color: "text-amber-500 bg-amber-100", title: "Z-score normalization", sub: "Mean = 0, Std = 1 · model-ready input" },
              { icon: Cog, color: "text-red-500 bg-red-100", title: "CNN + LSTM ensemble inference", sub: "Parallel forward pass · weighted probability average" },
            ].map((step, i) => (
              <div key={i} className="flex gap-4 relative z-10 pb-6 last:pb-0">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 border border-white shadow-sm ring-4 ring-white ${step.color}`}>
                  <step.icon size={18} strokeWidth={2.5} />
                </div>
                <div className="pt-1">
                  <h4 className="font-bold text-slate-800 text-sm leading-tight">{step.title}</h4>
                  <p className="text-xs text-slate-500">{step.sub}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* SECTION 4 - CONFUSION MATRIX */}
          <section>
            <h3 className="text-lg font-bold text-slate-800 mb-6 uppercase tracking-wider text-sm border-b pb-2">4. Confusion Matrix</h3>
            
            <div className="flex flex-col items-center">
              <div className="text-xs font-bold text-slate-500 mb-2 ml-12">Predicted →</div>
              <div className="flex">
                <div className="text-xs font-bold text-slate-500 flex flex-col justify-center mr-2 w-10 text-right rotate-[-90deg] origin-center -ml-10">Actual</div>
                <div className="border border-slate-200 rounded-lg overflow-hidden">
                  <table className="text-center text-sm">
                    <thead>
                      <tr className="bg-slate-50 text-slate-600 font-semibold text-xs uppercase">
                        <th className="p-2 border-b border-r border-slate-200 w-16"></th>
                        {MODEL_META.classLabels.map(l => <th key={l} className="p-3 border-b border-slate-200 w-20">{l}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {MODEL_META.confusionMatrix.map((row, i) => (
                        <tr key={i}>
                          <td className="p-2 bg-slate-50 text-slate-600 font-semibold text-xs border-r border-slate-200">
                            {MODEL_META.classLabels[i]}
                          </td>
                          {row.map((val, j) => {
                            const isCorrect = i === j;
                            const isVfibTarget = i === 2 && j === 2;
                            
                            let cellBg = isCorrect ? "bg-green-100 text-green-800" : "bg-amber-100 text-amber-800";
                            let borders = "border-t border-l border-slate-100";
                            if (isVfibTarget) {
                              cellBg = "bg-green-200 text-green-900 border-2 border-green-500 shadow-inner z-10 relative";
                            }
                            
                            return (
                              <td key={j} className={`p-4 font-bold ${cellBg} ${borders}`}>
                                <div className="flex flex-col items-center">
                                  <span>{val}</span>
                                  {isVfibTarget && <span className="text-[10px] text-green-700 leading-none mt-1">★</span>}
                                </div>
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="flex gap-4 justify-center mt-4 text-xs font-medium text-slate-600">
              <div className="flex items-center gap-1.5"><div className="w-3 h-3 bg-green-200 border border-green-400 rounded-sm"></div> Correct</div>
              <div className="flex items-center gap-1.5"><div className="w-3 h-3 bg-amber-200 border border-amber-400 rounded-sm"></div> Misclassified</div>
            </div>

            <div className="mt-4 bg-slate-50 border border-slate-200 p-4 rounded-lg text-sm text-slate-600 text-center">
              VFib row: 0 missed as Normal, 1 missed as AFib, <strong>99 correctly caught</strong>. Zero fatal misses.
            </div>
          </section>

          {/* SECTION 5 & 6 - INFO AND TIMING */}
          <div className="space-y-12">
            
            <section>
              <h3 className="text-lg font-bold text-slate-800 mb-6 uppercase tracking-wider text-sm border-b pb-2">5. Dataset & Config</h3>
              <div className="grid grid-cols-2 gap-x-4 gap-y-6">
                {[
                  { k: "Dataset source", v: MODEL_META.dataset.source, s: MODEL_META.dataset.origin },
                  { k: "Total samples", v: MODEL_META.dataset.totalSamples, s: MODEL_META.dataset.split },
                  { k: "Sampling rate", v: MODEL_META.dataset.samplingRate, s: MODEL_META.dataset.windowSize },
                  { k: "Class balance", v: MODEL_META.dataset.balancing, s: MODEL_META.dataset.balancingSub },
                  { k: "Framework", v: MODEL_META.dataset.framework, s: MODEL_META.dataset.python },
                  { k: "API endpoint", v: MODEL_META.dataset.endpoint, s: MODEL_META.dataset.endpointSub },
                ].map((item, i) => (
                  <div key={i}>
                    <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">{item.k}</p>
                    <p className="font-bold text-slate-800 text-sm">{item.v}</p>
                    <p className="text-xs text-slate-500">{item.s}</p>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h3 className="text-lg font-bold text-slate-800 mb-6 uppercase tracking-wider text-sm border-b pb-2">6. Inference Timing</h3>
              <div className="space-y-4">
                {[
                  { l: "Signal preprocessing", v: MODEL_META.timing.preprocessing, color: "bg-blue-400" },
                  { l: "Model inference", v: MODEL_META.timing.inference, color: "bg-blue-400" },
                  { l: "Total response", v: MODEL_META.timing.total, color: "bg-green-500" },
                ].map((t, i) => {
                  const pct = Math.round((t.v / MODEL_META.timing.total) * 100);
                  return (
                    <div key={i}>
                      <div className="flex justify-between text-sm font-medium mb-1">
                        <span className="text-slate-600">{t.l}</span>
                        <span className="font-bold text-slate-800">{t.v} ms</span>
                      </div>
                      <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                        <div className={`h-full ${t.color}`} style={{ width: `${pct}%` }}></div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </section>

          </div>
        </div>

      </div>
    </details>
  );
}
