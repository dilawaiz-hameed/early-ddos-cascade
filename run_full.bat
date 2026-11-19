@echo off
REM Full CIC-DDoS2019 run: train on 1 Dec 2018, test on 3 Nov 2018.
REM Results go to results_full\ and figures_full\ (the original results\ stay untouched).
set CN_MAIN=cicfull
set CN_RESULTS=results_full
set CN_FIGURES=figures_full
set OMP_NUM_THREADS=1

echo [1/5] Building flows from the capture files...
python src\flows_full.py "D:\Cnet project\PCAP-01-12_0-0249" "D:\Cnet project\PCAP-03-11" || exit /b 1
echo [2/5] Searching configurations (NSGA-II, random, GA)...
python src\run_search.py 0 1 2 || exit /b 1
echo [3/5] Running experiments...
python src\run_experiments.py || exit /b 1
echo [4/5] Measuring cost...
python src\cost_bench.py || exit /b 1
echo [5/5] Drawing figures...
python src\make_figures.py || exit /b 1
echo Done. Send results_full\experiments.json and results_full\cost_bench.json
