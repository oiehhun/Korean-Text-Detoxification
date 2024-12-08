### Classification 모델 학습(KcELECTRA) ###

# 1. 환경 설정
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from warnings import filterwarnings
filterwarnings("ignore")

## 1-1. GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)


# 2. 데이터셋 제작
## 2-1. 원본 데이터 불러오기
df = pd.read_csv("./datasets/merge_dataset.csv", sep=",")

## 2-2. 데이터 전처리
null_idx = df[df.lable.isnull()].index
df.loc[null_idx, "content"]

# lable 은 content의 가장 끝 문자열로 설정
df.loc[null_idx, "lable"] = df.loc[null_idx, "content"].apply(lambda x: x[-1])

# content는 "\t" 앞부분까지의 문자열로 설정
df.loc[null_idx, "content"] = df.loc[null_idx, "content"].apply(lambda x: x[:-2])

df = df.astype({"lable":"int"})
df.info()


## 2-3. Train set / Test set으로 나누기
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

# 데이터셋 갯수 확인
print('중복 제거 전 학습 데이터셋 : {}'.format(len(train_data)))
print('중복 제거 전 테스트 데이터셋 : {}'.format(len(test_data)))

# 중복 데이터 제거
train_data.drop_duplicates(subset=["content"], inplace= True)
test_data.drop_duplicates(subset=["content"], inplace= True)

# 데이터셋 갯수 확인
print('중복 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
print('중복 제거 후 테스트 데이터셋 : {}'.format(len(test_data)))


## 2-4. 토크나이징
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenized_train_sentences = tokenizer(
    list(train_data["content"]),
    return_tensors="pt",                # pytorch의 tensor 형태로 return
    max_length=128,                     # 최대 토큰길이 설정
    padding=True,                       # 제로패딩 설정
    truncation=True,                    # max_length 초과 토큰 truncate
    add_special_tokens=True,            # special token 추가
    )

print(tokenized_train_sentences[0])
print(tokenized_train_sentences[0].tokens)
print(tokenized_train_sentences[0].ids)
print(tokenized_train_sentences[0].attention_mask)

tokenized_test_sentences = tokenizer(
    list(test_data["content"]),
    return_tensors="pt",
    max_length=128,
    padding=True,
    truncation=True,
    add_special_tokens=True,
    )


## 2-5. 데이터셋 생성
class CurseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_label = train_data["lable"].values
test_label = test_data["lable"].values

train_dataset = CurseDataset(tokenized_train_sentences, train_label)
test_dataset = CurseDataset(tokenized_test_sentences, test_label)


# 3. 모델 학습
## 3-1. 모델 불러오기
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)


## 3-2. 학습 파라미터 설정
training_args = TrainingArguments(
    output_dir='./',                    # 학습결과 저장경로
    num_train_epochs=10,                # 학습 epoch 설정
    per_device_train_batch_size=8,      # train batch_size 설정
    per_device_eval_batch_size=64,      # test batch_size 설정
    logging_dir='./logs',               # 학습log 저장경로
    logging_steps=500,                  # 학습log 기록 단위
    save_total_limit=2,                 # 학습결과 저장 최대갯수 
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,                         # 학습하고자하는 🤗 Transformers model
    args=training_args,                  # 위에서 정의한 Training Arguments
    train_dataset=train_dataset,         # 학습 데이터셋
    eval_dataset=test_dataset,           # 평가 데이터셋
    compute_metrics=compute_metrics,     # 평가지표
)

## 3-3. 학습
trainer.train()


# 4. 모델 성능 평가
print(f'성능 평가 : {trainer.evaluate(eval_dataset=test_dataset)}')