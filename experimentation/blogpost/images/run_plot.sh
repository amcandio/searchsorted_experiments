python plot.py --csv ../results/all_results.csv --output figure1.png --include searchsorted_py
python plot.py --csv ../results/all_results.csv --output figure2.png --include searchsorted_py searchsorted_py_np
python plot.py --csv ../results/all_results.csv --output figure3.png --include searchsorted_py_np searchsorted_py_np_fixed
python plot.py --csv ../results/all_results.csv --output figure4.png --include searchsorted_py_np_fixed searchsorted_np_2.4
python plot.py --csv ../results/all_results.csv --output figure6.png --include searchsorted_py_np_fixed searchsorted_py_np_fast
python plot.py --csv ../results/all_results.csv --output figure_PR_reply.png --include searchsorted_np_2.5 searchsorted_py_np_fast

python plot.py --csv  ../results_cross/parallel_1_20k.csv --output cross_figure1.png
python plot.py --csv  ../results_cross/parallel_8_3_20k.csv --output cross_figure2.png
