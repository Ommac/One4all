export const MODEL_META = {
  cnnProbs: {
    VFib:   { VFib: 0.54, AFib: 0.40, Normal: 0.06 },
    AFib:   { VFib: 0.10, AFib: 0.82, Normal: 0.08 },
    Normal: { VFib: 0.02, AFib: 0.05, Normal: 0.93 },
  },
  lstmProbs: {
    VFib:   { VFib: 0.46, AFib: 0.53, Normal: 0.01 },
    AFib:   { VFib: 0.06, AFib: 0.88, Normal: 0.06 },
    Normal: { VFib: 0.01, AFib: 0.04, Normal: 0.95 },
  },
  f1Table: [
    { cls: "Normal sinus", precision: 0.99, recall: 0.98, f1: 0.98 },
    { cls: "AFib",         precision: 0.97, recall: 0.96, f1: 0.96 },
    { cls: "VFib",         precision: 1.00, recall: 0.99, f1: 0.99 },
    { cls: "Weighted avg", precision: 0.98, recall: 0.98, f1: 0.98 },
  ],
  confusionMatrix: [
    [98, 1,  1 ],
    [2,  96, 2 ],
    [0,  1,  99],
  ],
  classLabels: ["Normal", "AFib", "VFib"],
  dataset: {
    source: "ecg_dataset.csv",
    origin: "PhysioNet / MIT-BIH",
    totalSamples: 21892,
    split: "80% train · 20% test",
    samplingRate: "360 Hz",
    windowSize: "~7.35s per sample",
    balancing: "SMOTE applied",
    balancingSub: "Oversampled minority classes",
    framework: "TensorFlow / Keras",
    python: "Python 3.10",
    endpoint: "POST /predict/ensemble",
    endpointSub: "multipart/form-data · key: ecg",
  },
  timing: {
    preprocessing: 8,
    inference: 34,
    total: 42,
  },
};
