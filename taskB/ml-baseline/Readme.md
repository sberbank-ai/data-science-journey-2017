В данной папке содержится решение задачи B с использованием машинного обучения.

Идея решения: для каждый пары вопрос/параграф генерируем различных кандидатов на ответ и для каждого кандидата строим признаки, затем обучаем модель, которая будет предсказывать величину `F_1(candidate, real_answer)` (`candidate` - кандидат на ответ, `real answer` - правильный ответ на вопрос к данному параграфу, итоговым ответом является кандидат с максимальным предсказанным значением.

Для того чтобы решение можно было отправить в систему необходимо сохранить все сопутсвующие файлы:
1. `idf.pkl` - сохраненные IDF (inverse document frequency) веса слов по обучающей выборке
2. `model.pkl` - сохраненная модель для предсказания.
3. `taskB_ml_baseline_utils.py` - код для рассчета признаков

## Инструкция по созданию решения:

1. Склонируйте репозиторий себе на компьютер
2. Зайдите в папку `taskB/ml-baseline`, удалите `idf.pkl` и `model.pkl`, затем откройте `jupyter notebook` `train.ipynb` и выполните все ячейки, в результате у вас создастся два файла `idf.pkl` и `model.pkl`
2. Зайдите в папку `taskB`
3. Выполните команду: `python3 create_submission.py -p ml-baseline/model.pkl -p ml-baseline/model.pkl -p ml-baseline/taskB_ml_baseline_utils.py -p ml-baseline/predict.py -o output_ml.zip`)
4. Создастся архив: output_ml.zip, который можно отправить в систему


## Инструкция по проверке решения:

1. Склонируйте репозиторий себе на компьютер
2. Зайдите в папку `taskB`
3. Положите в папку `taskB` файл с данными (`train.csv`)
4. Выполните `python3 split_train.py train.csv`, в результате появится два файла: `train_without_validate.csv`, `validate.csv`. Первый будем в будущем использовать для обучения, второй для проверка работы моделей.
5. Зайдите в папку taskB/ml-baseline, удалите `idf.pkl` и `model.pkl`, затем откройте `jupyter notebook` `train.ipynb` и выполните все ячейки(поменяв название файла для обучение на `train_without_validate.csv`, в результате у вас создастся два файла `idf.pkl` и `model.pkl`
6. Зайдите в папку taskB
7. Выполните команду: `python3 create_submission.py -p ml-baseline/model.pkl -p ml-baseline/model.pkl -p ml-baseline/taskB_ml_baseline_utils.py -p ml-baseline/predict.py -o output_ml.zip`)
8. Выполните `python3 check_solution.py -t docker --submission_file output_ml.zip --data_file validate.csv`, в результате вы увидите в конце строчку вида: `{'f1': 0.3361121166011774}`
