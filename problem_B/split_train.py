import pandas as pd
from sklearn.utils import shuffle
import sys

if __name__ == '__main__':
    data = shuffle(pd.read_csv(sys.argv[1]), random_state=42)[['paragraph_id', 'question_id', 'paragraph', 'question', 'answer']]
    data.iloc[:10000].to_csv("validate.csv", header=True, index=False)
    data.iloc[10000:].to_csv("train_without_validate.csv", header=True, index=False)
