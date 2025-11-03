# Student-Alcohol-Consumption
Gen Ai Disclosure
Have used ChatGpt to generate ideas for motivation some parts of proposal and references

How to run baseline.py script
Clone the repo then
Windows (PowerShell)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\src\train_baselines.py --mat .\src\student-mat.csv --por .\src\student-por.csv --experiment midpointreport --run-name baseline-exp1


macOS/Linux (bash/zsh)

Create & activate venv:
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt


Run baseline:

python3 ./src/train_baselines.py \
  --mat ./src/student-mat.csv \
  --por ./src/student-por.csv \
  --experiment midpointreport \
  --run-name baseline-exp1


