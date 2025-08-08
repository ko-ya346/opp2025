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
papermill notebook/exp024.ipynb notebook/exp024_2_output.ipynb -p debug False 
