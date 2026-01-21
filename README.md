python RAW+KNN+SVM+LSTM+CNN.py --out .   --FUZZY_DATA_OUT data/toon_mhealth --train_jsonl data/toon_mhealth/mhealth_train_windows.jsonl   --test_jsonl  data/toon_mhealth/mhealth_test_windows.jsonl  --sensor_cfg config/mhealth.cfg --normalize
python RAW+KNN+SVM+LSTM+CNN.py --out .   --FUZZY_DATA_OUT data/toon_pamap --train_jsonl data/toon_pamap/pamap_train_windows.jsonl   --test_jsonl  data/toon_pamap/pamap_test_windows.jsonl  --sensor_cfg config/pamap.cfg --normalize
python RAW+KNN+SVM+LSTM+CNN.py --out .   --FUZZY_DATA_OUT data/toon_mobiact_adl --train_jsonl data/toon_mobiact_adl/mobiact_adl_train_windows.jsonl   --test_jsonl  data/toon_mobiact_adl/mobiact_adl_test_windows.jsonl  --sensor_cfg config/mobiact_adl.cfg

python RAW+KNN+SVM+LSTM+CNN.py --out .   --FUZZY_DATA_OUT data/toon_uwb --train_jsonl data/toon_uwb/har_train_windows.jsonl   --test_jsonl  data/toon_uwb/har_test_windows.jsonl  --sensor_cfg config/uwb.cfg
python RAW+KNN+SVM+LSTM+CNN.py --out .   --FUZZY_DATA_OUT data/toon_marble --train_jsonl data/toon_marble/marble_train_windows.jsonl   --test_jsonl  data/toon_marble/marble_test_windows.jsonl  --sensor_cfg config/marble.cfg
python RAW+KNN+SVM+LSTM+CNN.py --out .   --FUZZY_DATA_OUT data/toon_aruba --train_jsonl data/toon_aruba/aruba_train_bal_1000_46m.jsonl   --test_jsonl  data/toon_aruba/aruba_test_strat10_46m.jsonl  --sensor_cfg config/aruba.cfg



python Z.Simbolic+Stats.Trees.py --out .   --FUZZY_DATA_OUT data/toon_mhealth --train_jsonl data/toon_mhealth/mhealth_train_windows.jsonl   --test_jsonl  data/toon_mhealth/mhealth_test_windows.jsonl --sensor_cfg config/mhealth.cfg
python Z.Simbolic+Stats.Trees.py --out .   --FUZZY_DATA_OUT data/toon_pamap --train_jsonl data/toon_pamap/pamap_train_windows.jsonl   --test_jsonl  data/toon_pamap/pamap_test_windows.jsonl  --sensor_cfg config/pamap.cfg
python Z.Simbolic+Stats.Trees.py --out .   --FUZZY_DATA_OUT data/toon_mobiact_adl --train_jsonl data/toon_mobiact_adl/mobiact_adl_train_windows.jsonl   --test_jsonl  data/toon_mobiact_adl/mobiact_adl_test_windows.jsonl  --sensor_cfg config/mobiact_adl.cfg

python Z.Simbolic+Stats.Trees.py --out .   --FUZZY_DATA_OUT data/toon_marble --train_jsonl data/toon_marble/marble_train_windows.jsonl   --test_jsonl  data/toon_marble/marble_test_windows.jsonl  --sensor_cfg config/marble.cfg
python Z.Simbolic+Stats.Trees.py --out .   --FUZZY_DATA_OUT data/toon_uwb --train_jsonl data/toon_uwb/har_train_windows.jsonl   --test_jsonl  data/toon_uwb/har_test_windows.jsonl --sensor_cfg config/uwb.cfg
python Z.Simbolic+Stats.Trees.py --out .   --FUZZY_DATA_OUT data/toon_aruba --train_jsonl data/toon_aruba/aruba_train_bal_1000_46m.jsonl   --test_jsonl  data/toon_aruba/aruba_test_strat10_46m.jsonl  --sensor_cfg config/aruba.cfg


python -m src.training.llm_training --config config/mhealth_train2.cfg --config2 config/mhealth.cfg --device cuda
python -m src.training.llm_training --config config/pamap_train2.cfg --config2 config/pamap.cfg --device cuda
python -m src.training.llm_training --config config/mobiact_adl2.cfg --config2 config/mobiact_adl.cfg --device cuda

python -m src.training.llm_training --config config/marble_train2.cfg --config2 config/marble.cfg --device cuda
python -m src.training.llm_training --config config/uwb_train2.cfg --config2 config/uwb.cfg --device cuda
python -m src.training.llm_training --config config/aruba_train2.cfg --config2 config/aruba.cfg --device cuda




##############



