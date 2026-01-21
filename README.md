# 🧠 Symbolic + Interpretable + Models for Human Activity Recognition

This repository presents a **comparative and unified framework for Human Activity Recognition (HAR)** that combines:

* **Raw sensor-based models** (KNN, Linear SVM, LSTM, CNN)
* **Symbolic and statistical interpretable features** (fuzzy logic + statistics)
* **Classical interpretable ML models** (C4.5 decision trees, Linear SVM, XGBoost)
* **Tiny / Sensor-oriented Large Language Models (LLMs)** for reasoning and semantic interaction over sensor data

The goal of the project is to **systematically study the trade-off between performance and interpretability** in HAR, while enabling **symbolic explanations and LLM-based reasoning** grounded on sensor signals.

---

## 🔍 Main contributions

* Extraction of **symbolic fuzzy descriptors** from sensor time series
* Interpretable **statistical feature computation** (durations, transitions, peaks, variability)
* Unified evaluation of:

  * Raw-signal models (KNN, SVM, LSTM, CNN)
  * Symbolic + statistical models (C4.5, Linear SVM, XGBoost)
  * LLM-based HAR and explanation models
* Natural-language descriptions of sensor behaviour and activities
* Multi-dataset experimental pipeline

---

## 📊 Supported datasets

The framework has been evaluated on multiple **wearable, ambient, and hybrid HAR datasets**, including:

* **MHEALTH** – wearable IMU-based activities
* **PAMAP2** – multi-sensor physical activity monitoring
* **MobiAct (ADL)** – smartphone-based activities of daily living
* **MARBLE** – multi-resident wearable + ambient sensing
* **UWB HAR** – ultra-wideband indoor positioning-based activities
* **ARUBA** – ambient binary sensor smart-home dataset

Each dataset is preprocessed into **windowed JSONL files** and accompanied by a **sensor configuration file** describing channels, modalities, and symbolic partitions.

---

## 🚀 Running experiments

### 1️⃣ Raw-signal models (KNN, SVM, LSTM, CNN)

```bash
python RAW+KNN+SVM+LSTM+CNN.py --out . \
  --FUZZY_DATA_OUT data/toon_mhealth \
  --train_jsonl data/toon_mhealth/mhealth_train_windows.jsonl \
  --test_jsonl  data/toon_mhealth/mhealth_test_windows.jsonl \
  --sensor_cfg config/mhealth.cfg --normalize

python RAW+KNN+SVM+LSTM+CNN.py --out . \
  --FUZZY_DATA_OUT data/toon_pamap \
  --train_jsonl data/toon_pamap/pamap_train_windows.jsonl \
  --test_jsonl  data/toon_pamap/pamap_test_windows.jsonl \
  --sensor_cfg config/pamap.cfg --normalize

python RAW+KNN+SVM+LSTM+CNN.py --out . \
  --FUZZY_DATA_OUT data/toon_mobiact_adl \
  --train_jsonl data/toon_mobiact_adl/mobiact_adl_train_windows.jsonl \
  --test_jsonl  data/toon_mobiact_adl/mobiact_adl_test_windows.jsonl \
  --sensor_cfg config/mobiact_adl.cfg

python RAW+Models.py --out . \
  --FUZZY_DATA_OUT data/toon_uwb \
  --train_jsonl data/toon_uwb/har_train_windows.jsonl \
  --test_jsonl  data/toon_uwb/har_test_windows.jsonl \
  --sensor_cfg config/uwb.cfg

python RAW+Models.py --out . \
  --FUZZY_DATA_OUT data/toon_marble \
  --train_jsonl data/toon_marble/marble_train_windows.jsonl \
  --test_jsonl  data/toon_marble/marble_test_windows.jsonl \
  --sensor_cfg config/marble.cfg

python RAW+Models.py --out . \
  --FUZZY_DATA_OUT data/toon_aruba \
  --train_jsonl data/toon_aruba/aruba_train_bal_1000_46m.jsonl \
  --test_jsonl  data/toon_aruba/aruba_test_strat10_46m.jsonl \
  --sensor_cfg config/aruba.cfg
```

---

### 2️⃣ Symbolic + statistical models (C4.5, Linear SVM, XGBoost)

```bash
python Simbolic+Models.py --out . \
  --FUZZY_DATA_OUT data/toon_mhealth \
  --train_jsonl data/toon_mhealth/mhealth_train_windows.jsonl \
  --test_jsonl  data/toon_mhealth/mhealth_test_windows.jsonl \
  --sensor_cfg config/mhealth.cfg

python Simbolic+Models.py --out . \
  --FUZZY_DATA_OUT data/toon_pamap \
  --train_jsonl data/toon_pamap/pamap_train_windows.jsonl \
  --test_jsonl  data/toon_pamap/pamap_test_windows.jsonl \
  --sensor_cfg config/pamap.cfg

python Simbolic+Models.py --out . \
  --FUZZY_DATA_OUT data/toon_mobiact_adl \
  --train_jsonl data/toon_mobiact_adl/mobiact_adl_train_windows.jsonl \
  --test_jsonl  data/toon_mobiact_adl/mobiact_adl_test_windows.jsonl \
  --sensor_cfg config/mobiact_adl.cfg

python Simbolic+Models.py --out . \
  --FUZZY_DATA_OUT data/toon_marble \
  --train_jsonl data/toon_marble/marble_train_windows.jsonl \
  --test_jsonl  data/toon_marble/marble_test_windows.jsonl \
  --sensor_cfg config/marble.cfg

python Simbolic+Models.py --out . \
  --FUZZY_DATA_OUT data/toon_uwb \
  --train_jsonl data/toon_uwb/har_train_windows.jsonl \
  --test_jsonl  data/toon_uwb/har_test_windows.jsonl \
  --sensor_cfg config/uwb.cfg

python Simbolic+Models.py --out . \
  --FUZZY_DATA_OUT data/toon_aruba \
  --train_jsonl data/toon_aruba/aruba_train_bal_1000_46m.jsonl \
  --test_jsonl  data/toon_aruba/aruba_test_strat10_46m.jsonl \
  --sensor_cfg config/aruba.cfg
```

---

### 3️⃣ LLM-based HAR training

```bash
python -m src.training.llm_training --config config/mhealth_train2.cfg --config2 config/mhealth.cfg --device cuda
python -m src.training.llm_training --config config/pamap_train2.cfg --config2 config/pamap.cfg --device cuda
python -m src.training.llm_training --config config/mobiact_adl2.cfg --config2 config/mobiact_adl.cfg --device cuda

python -m src.training.llm_training --config config/marble_train2.cfg --config2 config/marble.cfg --device cuda
python -m src.training.llm_training --config config/uwb_train2.cfg --config2 config/uwb.cfg --device cuda
python -m src.training.llm_training --config config/aruba_train2.cfg --config2 config/aruba.cfg --device cuda
```

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

Pretrained models and datasets are subject to their respective original licenses.

---

## 📚 Citation

If you use this repository in academic work, please cite the corresponding publication (details to be added).
