import React, { useRef } from 'react';
import { UploadCloud, FileText, Loader2 } from 'lucide-react';

export default function UploadPanel({ onAnalyze, isLoading, fileName }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onAnalyze(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      onAnalyze(file);
    }
  };

  return (
    <div className="bg-white border-2 border-slate-200 rounded-xl p-6 shadow-sm mb-6 flex flex-col md:flex-row items-center gap-6">
      <div 
        className="flex-1 w-full border-2 border-dashed border-slate-300 rounded-xl p-8 hover:bg-slate-50 hover:border-slate-400 transition-colors cursor-pointer flex flex-col items-center justify-center text-center"
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <input 
          type="file" 
          accept=".csv,.json" 
          className="hidden" 
          ref={fileInputRef} 
          onChange={handleFileChange}
        />
        <UploadCloud className="text-slate-400 mb-3" size={40} />
        <h3 className="font-bold text-slate-800 text-lg mb-1">
          Upload ECG file (.csv or .json)
        </h3>
        <p className="text-slate-500 text-sm font-medium">Drag & drop or click to browse</p>
      </div>

      <div className="flex-1 w-full flex flex-col justify-center gap-4">
        {fileName ? (
          <div className="bg-emerald-50 text-emerald-900 border border-emerald-200 px-4 py-3 rounded-lg flex items-center gap-3">
            <FileText className="text-emerald-600" size={24} />
            <div className="overflow-hidden">
              <p className="font-bold text-sm truncate">{fileName}</p>
              <p className="text-xs text-emerald-700/80 font-medium">Ready for review</p>
            </div>
          </div>
        ) : (
          <div className="bg-slate-50 text-slate-500 border border-slate-200 px-4 py-3 rounded-lg flex items-center justify-center h-[74px]">
            <p className="font-medium text-sm text-center">No patient file active</p>
          </div>
        )}

        <button 
          onClick={() => document.querySelector('input[type="file"]').click()}
          disabled={isLoading}
          className="w-full bg-[#1e293b] text-white hover:bg-[#0f172a] disabled:opacity-50 disabled:cursor-not-allowed font-bold text-lg py-4 px-6 rounded-xl flex items-center justify-center gap-3 transition-colors shadow-sm"
        >
          {isLoading ? (
            <>
              <Loader2 className="animate-spin" size={24} />
              Processing...
            </>
          ) : (
            'Analyze'
          )}
        </button>
      </div>
    </div>
  );
}
