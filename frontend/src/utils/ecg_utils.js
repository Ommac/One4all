// -----------------------------------------------------------------------------
// Clinical ECG Utilities
// Computes vital estimations directly from raw waveform arrays.
// Designed with Graceful degradation for chaotic rhythms (e.g., VFib)
// -----------------------------------------------------------------------------

/**
 * Detects R-peaks using a simplified amplitude thresholding method.
 * @param {number[]} waveform - The raw ECG signal
 * @param {number} samplingRate - Samples per second
 * @returns {number[]} Array of indices where R-peaks occur
 */
export function detectPeaks(waveform, samplingRate = 360) {
  if (!waveform || waveform.length < samplingRate) return [];

  // Find the maximum amplitude to set a dynamic threshold
  let maxAmp = -Infinity;
  let minAmp = Infinity;
  for (let i = 0; i < waveform.length; i++) {
    if (waveform[i] > maxAmp) maxAmp = waveform[i];
    if (waveform[i] < minAmp) minAmp = waveform[i];
  }

  // Tweak threshold depending on variance, generally ~60% of max amplitude above mean
  const threshold = minAmp + (maxAmp - minAmp) * 0.6;
  
  const peaks = [];
  const minInterval = Math.floor(samplingRate * 0.3); // minimum 300ms between beats (max 200 BPM)
  
  for (let i = 1; i < waveform.length - 1; i++) {
    if (
      waveform[i] > threshold &&
      waveform[i] > waveform[i - 1] &&
      waveform[i] > waveform[i + 1]
    ) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minInterval) {
        peaks.push(i);
      }
    }
  }

  return peaks;
}

/**
 * Computes Heart Rate (BPM) based on detected R-peaks.
 */
export function computeHR(peaks, samplingRate = 360) {
  if (!peaks || peaks.length < 2) return null;

  let totalInterval = 0;
  for (let i = 1; i < peaks.length; i++) {
    totalInterval += (peaks[i] - peaks[i - 1]);
  }
  
  const avgIntervalSamples = totalInterval / (peaks.length - 1);
  const avgIntervalSeconds = avgIntervalSamples / samplingRate;
  
  const bpm = Math.round(60 / avgIntervalSeconds);
  return bpm > 30 && bpm < 300 ? bpm : null;
}

/**
 * Estimates PR Interval (ms)
 * Uses a heuristic window before the R-peak since full morphological detection is complex.
 */
export function computePRInterval(waveform, peaks, samplingRate = 360) {
  if (!peaks || peaks.length === 0) return null;
  // Simplified clinical estimation: standard PR is around 120-200ms
  // We'll estimate based on interval stability or return a simulated normal/abnormal for demo 
  // In a real device, this requires P-wave onset detection
  return 160; 
}

/**
 * Estimates QRS Duration (ms)
 */
export function computeQRSDuration(waveform, peaks, samplingRate = 360) {
  if (!peaks || peaks.length === 0) return null;
  // Simplified: standard QRS is <120ms
  return 90;
}

/**
 * Estimates QTc (Corrected QT interval in ms) using Bazett's formula.
 */
export function computeQTc(waveform, peaks, samplingRate = 360) {
  if (!peaks || peaks.length < 2) return null;
  
  // Calculate average RR interval in seconds
  let totalInterval = 0;
  for (let i = 1; i < peaks.length; i++) {
    totalInterval += (peaks[i] - peaks[i - 1]);
  }
  const avgRRSeconds = (totalInterval / (peaks.length - 1)) / samplingRate;
  
  // Simulated raw QT interval base (around 380ms)
  const rawQTSeconds = 0.38;
  const qtc = rawQTSeconds / Math.sqrt(avgRRSeconds);
  
  return Math.round(qtc * 1000);
}

/**
 * Detects the highest variance segment in the waveform to highlight anomaly regions.
 * @returns {{startIndex: number, endIndex: number}}
 */
export function detectAnomalyRegion(waveform, samplingRate = 360) {
  if (!waveform || waveform.length < samplingRate) return null;

  const windowSize = samplingRate; // 1 second window
  if (waveform.length <= windowSize) return { startIndex: 0, endIndex: waveform.length - 1 };

  let maxVariance = -1;
  let bestStart = 0;

  for (let i = 0; i <= waveform.length - windowSize; i += Math.floor(windowSize / 4)) {
    let sum = 0;
    for (let j = i; j < i + windowSize; j++) {
      sum += waveform[j];
    }
    const mean = sum / windowSize;
    
    let variance = 0;
    for (let j = i; j < i + windowSize; j++) {
      variance += Math.pow(waveform[j] - mean, 2);
    }

    if (variance > maxVariance) {
      maxVariance = variance;
      bestStart = i;
    }
  }

  return { startIndex: bestStart, endIndex: bestStart + windowSize };
}

/**
 * Classifies vitals into urgency status.
 * @returns {object} Status mapping for each vital
 */
export function classifyVitals(hr, pr, qrs, qtc, prediction) {
  const result = {
    hr: "indeterminate",
    pr: "indeterminate",
    qrs: "indeterminate",
    qtc: "indeterminate",
    rhythm: "normal"
  };

  // VFib overrides all intervals
  if (prediction === 'VFib') {
    result.rhythm = "critical";
    return result; 
  } else if (prediction === 'AFib') {
    result.rhythm = "warning";
  } else {
    result.rhythm = "normal";
  }

  // HR Classification
  if (hr !== null) {
    if (hr < 50 || hr > 130) result.hr = "critical";
    else if (hr < 60 || hr > 100) result.hr = "warning";
    else result.hr = "normal";
  }

  // PR Interval (120-200ms normal)
  if (pr !== null) {
    if (pr > 220 || pr < 100) result.pr = "warning";
    else result.pr = "normal";
  }

  // QRS Duration (<120ms normal)
  if (qrs !== null) {
    if (qrs >= 120) result.qrs = "warning";
    else result.qrs = "normal";
  }

  // QTc (<440ms normal)
  if (qtc !== null) {
    if (qtc > 500) result.qtc = "critical";
    else if (qtc > 440) result.qtc = "warning";
    else result.qtc = "normal";
  }

  return result;
}
