import React, { useRef, useState } from 'react';
import { UploadCloud, FileText, Loader2, CheckCircle } from 'lucide-react';

export default function UploadPanel({ onAnalyze, isLoading, fileName }) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onAnalyze(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onAnalyze(file);
  };

  return (
    <div className="animate-float-in glass-card p-6 md:p-8" id="upload-panel">
      <div className="flex flex-col md:flex-row items-stretch gap-6">
        
        {/* Drop Zone */}
        <div 
          className={`
            flex-1 border-2 border-dashed rounded-2xl p-8 
            flex flex-col items-center justify-center text-center cursor-pointer
            transition-all duration-300
            ${isDragOver 
              ? 'border-cyan-400/60 bg-cyan-500/5 shadow-lg shadow-cyan-500/10' 
              : 'border-slate-600/40 hover:border-slate-500/60 hover:bg-slate-800/30'
            }
          `}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          <input 
            type="file" 
            accept=".csv,.json" 
            className="hidden" 
            ref={fileInputRef} 
            onChange={handleFileChange}
          />

          <div className={`
            w-16 h-16 rounded-2xl mb-4 flex items-center justify-center
            transition-all duration-300
            ${isDragOver 
              ? 'bg-cyan-500/15 scale-110' 
              : 'bg-slate-700/50'
            }
          `}>
            <UploadCloud 
              className={`transition-colors duration-300 ${isDragOver ? 'text-cyan-400' : 'text-slate-400'}`} 
              size={32} 
            />
          </div>

          <h3 className="font-bold text-white text-lg mb-1">
            Upload ECG file (.csv or .json)
          </h3>
          <p className="text-slate-500 text-sm font-medium">
            Drag & drop or click to browse
          </p>
        </div>

        {/* Status + Analyze */}
        <div className="flex-1 flex flex-col justify-center gap-4">
          
          {/* File status */}
          {fileName ? (
            <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 px-5 py-4 rounded-xl flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/15 flex items-center justify-center flex-shrink-0">
                <CheckCircle className="text-emerald-400" size={20} />
              </div>
              <div className="overflow-hidden">
                <p className="font-bold text-sm truncate text-white">{fileName}</p>
                <p className="text-xs text-emerald-400/80 font-medium">Ready for analysis</p>
              </div>
            </div>
          ) : (
            <div className="bg-slate-800/40 border border-slate-700/40 text-slate-500 px-5 py-4 rounded-xl flex items-center justify-center" style={{ minHeight: 72 }}>
              <div className="flex items-center gap-2">
                <FileText size={18} className="text-slate-600" />
                <p className="font-medium text-sm">No patient file active</p>
              </div>
            </div>
          )}

          {/* Analyze button */}
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="
              w-full font-bold text-lg py-4 px-6 rounded-xl 
              flex items-center justify-center gap-3 
              transition-all duration-300
              disabled:opacity-50 disabled:cursor-not-allowed
              bg-gradient-to-r from-cyan-500 to-blue-600
              hover:from-cyan-400 hover:to-blue-500
              text-white shadow-lg shadow-cyan-500/20
              hover:shadow-xl hover:shadow-cyan-500/30
              active:scale-[0.98]
            "
            id="analyze-button"
          >
            {isLoading ? (
              <>
                <Loader2 className="animate-spin" size={22} />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Activity size={22} />
                <span>Analyze</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

function Activity({ size, ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}
