import React, { useState } from 'react';
import AlertBanner from './components/AlertBanner';
import ActionPanel from './components/ActionPanel';
import VitalsStrip from './components/VitalsStrip';
import ECGWaveform from './components/ECGWaveform';
import DifferentialDx from './components/DifferentialDx';
import UploadPanel from './components/UploadPanel';
import TechDetails from './components/TechDetails';
import { 
  detectPeaks, computeHR, computePRInterval, computeQRSDuration, 
  computeQTc, detectAnomalyRegion, classifyVitals 
} from './utils/ecg_utils';

const API_URL = 'http://localhost:5000';

export default function App() {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  const [vitals, setVitals] = useState({
    hr: null, pr: null, qrs: null, qtc: null, status: {}
  });
  const [anomalyRegion, setAnomalyRegion] = useState(null);

  const handleAnalyze = async (selectedFile) => {
    setFile(selectedFile);
    setError(null);
    setResult(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('ecg', selectedFile);

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Server error');
      }

      setResult(data);

      if (data.signal && Array.isArray(data.signal)) {
        const peaks = detectPeaks(data.signal);
        const hr = computeHR(peaks);
        const pr = computePRInterval(data.signal, peaks);
        const qrs = computeQRSDuration(data.signal, peaks);
        const qtc = computeQTc(data.signal, peaks);
        const status = classifyVitals(hr, pr, qrs, qtc, data.prediction);
        const region = detectAnomalyRegion(data.signal);

        setVitals({ hr, pr, qrs, qtc, status });
        setAnomalyRegion(region);
      }
      
    } catch (err) {
      setError(err.message || 'Failed to analyze ECG');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 p-4 md:p-8 font-sans text-slate-800">
      <div className="max-w-6xl mx-auto space-y-6">
        
        <header className="mb-8">
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Cardiac Monitoring Dashboard</h1>
          <p className="text-slate-500 font-semibold mt-1">Clinical Decision Support System</p>
        </header>

        {result?.prediction && <AlertBanner prediction={result.prediction} />}
        {result?.prediction && <ActionPanel prediction={result.prediction} />}

        {error && (
            <div className="bg-red-50 text-red-600 border border-red-200 p-4 rounded-lg font-bold">
              {error}
            </div>
        )}

        <UploadPanel 
          onAnalyze={handleAnalyze} 
          isLoading={isLoading} 
          fileName={file ? file.name : null} 
        />

        {result && (
          <>
            <VitalsStrip 
              hr={vitals.hr} 
              pr={vitals.pr} 
              qrs={vitals.qrs} 
              qtc={vitals.qtc} 
              prediction={result.prediction} 
              status={vitals.status} 
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <ECGWaveform data={result.signal} anomalyRegion={anomalyRegion} />
              </div>
              <div className="lg:col-span-1">
                <DifferentialDx probabilities={result.probabilities} />
              </div>
            </div>

            <TechDetails 
              prediction={result.prediction} 
              confidence={result.confidence} 
              probabilities={result.probabilities} 
            />
          </>
        )}

      </div>
    </div>
  );
}
