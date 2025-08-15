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
papermill notebook/exp030.ipynb notebook/executed/exp030_output.ipynb -p debug False 
