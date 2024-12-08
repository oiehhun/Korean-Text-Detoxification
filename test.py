### End-to-End Test ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'
import warnings
warnings.filterwarnings('ignore')

import shap
import scipy as sp

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

llm = OpenAI()
davinci3= OpenAI(
    model_name = 'text-davinci-003',
    max_tokens = 1000,
    temperature = 0
)

# ============================================< Classification >============================================

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 토크나이징
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 모델 생성
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# 모델 불러오기
model.load_state_dict(torch.load('./KcELECTRA_5_pth/checkpoint-4500/pytorch_model.bin'))

# 모델 예측
# 0: curse, 1: non_curse
def sentence_predict(sent):
    # 평가모드로 변경
    model.eval()

    # 입력된 문장 토크나이징
    tokenized_sent = tokenizer(
        sent,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        max_length=128
    )
    
    # 모델이 위치한 GPU로 이동 
    tokenized_sent.to(device)

    # 예측
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_sent["input_ids"],
            attention_mask=tokenized_sent["attention_mask"],
            token_type_ids=tokenized_sent["token_type_ids"]
            )

    # 결과 return
    logits = outputs[0]
    logits = logits.detach().cpu()
    result = logits.argmax(-1)
    if result == 0:
        result = " >> 악성댓글 👿"
    elif result == 1:
        result = " >> 정상댓글 😀"
    return result

# 댓글 입력
samples = pd.read_csv('./datasets/fianl_toxic_dataset.csv', sep=',')
samples = [sample.strip() for sample in samples]
samples = pd.DataFrame(samples, columns=['content'])

# 불용어 제거
file_path = './datasets/stopwords.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    # 파일 내용을 줄 단위로 읽어와 리스트에 저장
    lines = file.readlines()
stopwords = [line.strip() for line in lines]


# ============================================< XAI >============================================
# SHAP
# define a prediction function
def f(x):
    tv = torch.tensor([tokenizer.encode(v, pad_to_max_length=True, max_length=128, truncation=True) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

# build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer)


# ============================================< Masking >============================================
original = []
generate = []

for sentence in samples['content']:

    ori = sentence # 원본 문장
    original.append(ori)

    shap_values = explainer([sentence])

    threshold = sum([i for i in shap_values[0].values if i < 0]) / len([i for i in shap_values[0].values if i < 0])
    
    # 마스킹
    mask_list = []
    shap_values_zip = zip(shap_values[0].values, shap_values[0].data)
    for shap_value, feature in shap_values_zip:
        if shap_value < threshold: # threshold
            if feature.strip() not in stopwords: # stopwords 제거
                if feature.strip().isalnum(): # 특수문자 제거
                    mask_list.append(feature.strip())
    print('-'*100)
    print('masking list:', mask_list)
    print(f"원본 문장 : {sentence}")

    for mask in mask_list:
        sentence = sentence.replace(mask, "[mask]")

    print(f"마스킹 문장 : {sentence}")

    ori_mask = sentence # 마스킹된 문장


    # ============================================ < LLM > ============================================

    examples = [
        {
            "sentence" : "노령연금 받는 틀딱들이, 세금내는 개돼지들보다 많아서 의문이다.",
            "masking_sentence" : "노령연금 받는 [mask], 세금내는 [mask]보다 많아서 의문이다.",
            "answer":
        '''
        노령연금 받는 노인들이, 세금내는 사람들보다 많아서 의문이다.
        '''
        },
        {
            "sentence" : "좌빨 영화 납시요 개돼지들 선동시키기 딱이요",
            "masking_sentence" : "[mask] 영화 납시요 [mask] [mask]시키기 딱이요",
            "answer":
        '''
        좌파 영화 납시요. 사람들 부추기기 딱이요 
        '''
        },
        {
            "sentence" : "음주운전하는 새끼들은 진짜 대가리에 뭐가 든건지... 다 무기징역 시켜라",
            "masking_sentence" : "음주운전하는 [mask] 진짜 [mask] 뭐가 든건지... 다 무기징역 시켜라",
            "answer":
        '''
        음주운전하는 사람들은 진짜 머리에 뭐가 든건지... 다 무기징역 시켜라
        '''
        },
        {
            "sentence" : "대깨문이 문재인 협박범을 쉴드치네? 역시 대가리가 붕어인듯~",
            "masking_sentence" : "[mask] 문재인 협박범을 쉴드치네? 역시 [mask] [mask]~",
            "answer":
        '''
        문재인지지자들이 문재인 협박범을 쉴드치네? 역시 머리가 나쁜 듯~
        '''
        }
    ]

    template = """너는 문장 스타일 변환을 하는 역할을 할거야. 
    원래 댓글과, 해당 댓글이 악성 댓글로 판단되는데 중요한 영향을 미친(=feature importance가 높은) 단어를 masking 처리한 댓글이 주어지면 기존 댓글의 context는 유지하면서 주어진 스타일에 맞는 댓글을 생성하는 것이 너의 임무야.
    
    sentence : {sentence} 
    masking_sentence : {masking_sentence}
    생성 댓글: {answer}
    """

    example_prompt = PromptTemplate(
        input_variables=["sentence", "masking_sentence", "answer"], 
        template=template)

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="sentence: {origin}\nmasking_sentence: {masked}",
        input_variables=["origin", "masked"]
    )

    result = davinci3(
        prompt.format(origin=ori, masked =ori_mask)
    )
    print(result)

    gener = result.split('\n')[-1].strip()
    generate.append(gener)

df = pd.DataFrame({'original':original, 'generate':generate})
df.to_csv('./datasets/test_result.csv', index=False)