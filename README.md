# UAV CNN Robustness Experiment

Проект досліджує стійкість YOLO до:
- smoke
- dust
- motion blur
- combined degradation

## Встановлення

pip install -r requirements.txt

## Навчання

python scripts/train.py

## Генерація завад

python scripts/generate_degradations.py

## Оцінка

python scripts/evaluate.py