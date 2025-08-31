# 素朴なgnn, grea
# jupyter nbconvert --to notebook --execute ./notebook/exp018.ipynb
# jupyter nbconvert --to notebook --execute ./notebook/exp020.ipynb
# jupyter nbconvert --to notebook --execute ./notebook/exp022.ipynb
# jupyter nbconvert --to notebook --execute ./notebook/exp023.ipynb
# papermill notebook/exp023.ipynb notebook/exp023_1_output.ipynb -p exp "exp023_1" -p augmented_feature '["morgan", "maccs"]'
# papermill notebook/exp023.ipynb notebook/exp023_2_output.ipynb -p exp "exp023_2" -p augmented_feature '["morgan"]'
# papermill notebook/exp023.ipynb notebook/exp023_3_output.ipynb -p exp "exp023_3" -p augmented_feature '[]'
# papermill notebook/exp023.ipynb notebook/exp023_4_output.ipynb -p exp "exp023_4" -p drop_ratio 0.3 
# papermill notebook/exp023.ipynb notebook/exp023_5_output.ipynb -p exp "exp023_5" -p drop_ratio 0.1
# papermill notebook/exp024.ipynb notebook/exp024_1_output.ipynb -p debug False 

# トリマー環状化した smiles で学習
# papermill notebook/exp025.ipynb notebook/exp025_output.ipynb -p debug False 
# mordred 特徴量を混ぜた lightgbm
# papermill notebook/exp026.ipynb notebook/exp026_output.ipynb -p debug False 
# target 毎に学習
# papermill notebook/exp027.ipynb notebook/executed/exp027_output.ipynb -p debug False 
# 3d の距離とかを特徴量に追加
# papermill notebook/exp028.ipynb notebook/exp028_output.ipynb -p debug False 
# シンプルlightgbm
# papermill notebook/exp020.ipynb notebook/executed/exp020_output.ipynb -p debug False 

# scaffold で groupkfold
# papermill notebook/exp029.ipynb notebook/executed/exp029_output.ipynb -p debug False 
# exp027 を groupkfold に変更, torch-molecule のモデル全般試す
# papermill notebook/exp030.ipynb notebook/executed/exp030_output.ipynb -p debug False 

# papermill notebook/exp035.ipynb notebook/executed/exp035_output_1.ipynb -p debug False -p notes "control" -p is_trimmer_cyclic False -p exp "exp035-1"
# papermill notebook/exp035.ipynb notebook/executed/exp035_output_2.ipynb -p debug False -p notes "graph pooling=mean" -p exp "exp035-2" -p is_trimmer_cyclic False -p graph_pooling "mean"
# papermill notebook/exp035.ipynb notebook/executed/exp035_output_3.ipynb -p debug False -p notes "use desc" -p exp "exp035-3" -p is_trimmer_cyclic False -p augmented_feature "['morgan', 'maccs', 'desc']" 
# papermill notebook/exp035.ipynb notebook/executed/exp035_output_4.ipynb -p debug False -p notes "is_trimmer_cyclic" -p exp "exp035-4" -p is_trimmer_cyclic True 
# papermill notebook/exp035.ipynb notebook/executed/exp035_output_5.ipynb -p debug False -p notes "is_trimmer_cyclic, desc, graph pooling mean" -p is_trimmer_cyclic True -p exp "exp035-5" -p graph_pooling "mean" -p augmented_feature "['morgan', 'maccs', 'desc']"
# papermill notebook/exp035.ipynb notebook/executed/exp035_output_6.ipynb -p debug False -p notes "batch_size 256" -p exp "exp035-6" -p is_trimmer_cyclic False -p batch_size 256
# papermill notebook/exp035.ipynb notebook/executed/exp035_output_7.ipynb -p debug False -p notes "num epoch 10000" -p exp "exp035-7" -p is_trimmer_cyclic False -p num_epochs 10000
# papermill notebook/exp035.ipynb notebook/executed/exp035_output_8.ipynb -p debug False -p notes "drop ratio 0.3" -p exp "exp035-8" -p is_trimmer_cyclic False -p drop_ratio 0.3
# papermill notebook/exp035.ipynb notebook/executed/exp035_output_9.ipynb -p debug False -p notes "drop ratio 0.1" -p exp "exp035-9" -p is_trimmer_cyclic False -p drop_ratio 0.1
# papermill notebook/exp036.ipynb notebook/executed/exp036_output_1.ipynb -p debug False -p notes "fold 切り替え変更" -p exp "exp036-1"
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_1.ipynb -p notes "control" -p exp "exp037-2" -p set_seed True -p debug False
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_2.ipynb -p notes "control" -p exp "exp037-2" -p set_seed True -p debug False
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_12.ipynb -p notes "control" -p exp "exp037-2" -p set_seed True -p debug False
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_3.ipynb -p notes "use desc1" -p exp "exp037-3" -p set_seed True -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" 
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_4.ipynb -p notes "use desc2" -p exp "exp037-4" -p set_seed True -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" 
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_5.ipynb -p notes "use desc3" -p exp "exp037-5" -p set_seed True -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" 
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_6.ipynb -p notes "use desc & graph pooling mean1" -p exp "exp037-6" -p set_seed True -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" -p graph_pooling "mean"
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_7.ipynb -p notes "use desc & graph pooling mean2" -p exp "exp037-7" -p set_seed True -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" -p graph_pooling "mean"
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_8.ipynb -p notes "use desc & graph pooling mean3" -p exp "exp037-8" -p set_seed True -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" -p graph_pooling "mean"
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_9.ipynb -p notes "use desc & graph pooling mean & drop_ratio 1" -p exp "exp037-9" -p set_seed True -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" -p graph_pooling "mean" -p drop_ratio 0.1
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_10.ipynb -p notes "use desc & graph pooling mean & drop_ratio 2" -p exp "exp037-10" -p set_seed True -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" -p graph_pooling "mean" -p drop_ratio 0.1
# papermill notebook/exp037.ipynb notebook/executed/exp037_output_11.ipynb -p notes "use desc & graph pooling mean & drop_ratio 3" -p exp "exp037-11" -p set_seed True -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" -p graph_pooling "mean" -p drop_ratio 0.1
# papermill notebook/exp046.ipynb notebook/executed/exp046_output_1.ipynb
# papermill notebook/exp047.ipynb notebook/executed/exp047_output_1.ipynb -p notes "cv 調整" -p exp "exp047-1" -p debug False -p augmented_feature "['morgan', 'maccs']" -p graph_pooling "mean"
# papermill notebook/exp047.ipynb notebook/executed/exp047_output_2.ipynb -p notes "cv 調整 & 三量体" -p exp "exp047-2" -p debug False -p augmented_feature "['morgan', 'maccs']" -p graph_pooling "mean" -p is_trimmer_cyclic True
# papermill notebook/exp047.ipynb notebook/executed/exp047_output_3.ipynb -p notes "cv 調整 & desc" -p exp "exp047-3" -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" -p graph_pooling "mean"

# papermill notebook/exp049.ipynb notebook/executed/exp049.ipynb -p debug False
papermill notebook/exp049.ipynb notebook/executed/exp049-1.ipynb -p debug False
# papermill notebook/exp047.ipynb notebook/executed/exp047_output_4.ipynb -p notes "cv 調整 & 三量体& 三量体失敗は除外" -p exp "exp047-4" -p debug False -p augmented_feature "['morgan', 'maccs']" -p graph_pooling "mean" -p is_trimmer_cyclic True -p ignore_error_asterisk True
# papermill notebook/exp047.ipynb notebook/executed/exp047_output_5.ipynb -p notes "cv 調整 & desc & アスタリスク2つ以外は除外" -p exp "exp047-5" -p debug False -p augmented_feature "['morgan', 'maccs', 'desc']" -p graph_pooling "mean" -p is_trimmer_cyclic False -p ignore_error_asterisk True
# papermill notebook/exp047.ipynb notebook/executed/exp047_output_6.ipynb -p notes "cv 調整 & augmented_featureなし & アスタリスク2つ以外削除" -p exp "exp047-6" -p debug False -p augmented_feature "[]" -p graph_pooling "mean" -p is_trimmer_cyclic False -p ignore_error_asterisk True
# papermill notebook/exp047.ipynb notebook/executed/exp047_output_7.ipynb -p notes "cv 調整 & augmented_featureなし&三量体 & アスタリスク2つ以外削除" -p exp "exp047-7" -p debug False -p augmented_feature "[]" -p graph_pooling "mean" -p is_trimmer_cyclic True -p ignore_error_asterisk True

# python3 ./scripts/train_gnn.py exp047 exp047-1 -graph_pooling mean -augmented_feature morgan maccs
# python3 ./scripts/train_gnn.py exp047 exp047-2 -graph_pooling mean -augmented_feature morgan maccs -is_trimmer_cyclic
# python3 ./scripts/train_gnn.py exp047 exp047-3 -graph_pooling mean -augmented_feature morgan maccs desc 
# python3 ./scripts/train_gnn.py exp047 exp047-4 -graph_pooling mean -augmented_feature morgan maccs -is_trimmer_cyclic -ignore_error_asterisk
# python3 ./scripts/train_gnn.py exp047 exp047-5 -graph_pooling mean -augmented_feature morgan maccs desc -ignore_error_asterisk 
# python3 ./scripts/train_gnn.py exp047 exp047-6 -graph_pooling mean -ignore_error_asterisk 
# python3 ./scripts/train_gnn.py exp047 exp047-7 -graph_pooling mean -is_trimmer_cyclic -ignore_error_asterisk 
papermill notebook/exp048.ipynb notebook/executed/exp048-1.ipynb -p debug False -p notes "lgb control" -p exp exp048-1 -p ignore_3D False -p ignore_3d_stats True
papermill notebook/exp048.ipynb notebook/executed/exp048-2.ipynb -p debug False -p notes "use 3d stats" -p exp exp048-2 -p ignore_3D False -p ignore_3d_stats False
papermill notebook/exp048.ipynb notebook/executed/exp048-3.ipynb -p debug False -p notes "no use 3d" -p exp exp048-3 -p ignore_3D True -p ignore_3d_stats True

python3 ./scripts/train_lgb.py exp046 exp046 
python3 ./scripts/train_lgb.py exp048 exp048
