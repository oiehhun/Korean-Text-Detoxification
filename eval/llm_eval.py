import numpy as np
import pandas as pd

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ============================================< LLM Evaluation >============================================

llm = OpenAI()
davinci3= OpenAI(
    model_name = 'text-davinci-003',
    max_tokens = 1000,
    temperature = 0
)

examples = [
    {
        'sentence': '노령연금 받는 틀딱들이, 세금내는 개돼지들보다 많아서 의문이다.',
        'answer': '노령연금 받는 노인들이, 세금내는 사람들보다 많아서 의문이다.',
        'score1': 5.0,
        'score2': 5.0,
        'score3': 5.0
    },
    {
        'sentence': '대가리를 밟혀봐야 정신드는 유형',
        'answer': '사람들을 밟혀봐야 정신드는 유형',
        'score1': 5.0,
        'score2': 3.0,
        'score3': 1.0
    }
]

template = """
너는 문장 스타일 변환의 성능을 측정하는 역할로써 점수를 매겨 생성을 하는 것이 너의 임무야.

첫번째 : 스타일이 얼마나 잘 변환됐는지 확인(문장 S2와 비교해서 문장 S1이 얼마나 달라졌는지 1점부터(완전히 똑같은 스타일) 5점까지(완전히 다른 스타일) 점수를 매겨)

두번째 : 내용이 얼마나 잘 유지되는지 확인(문장 S2와 비교해서 문장 S1이 얼마나 내용을 잘 유지하고 있는지 0점부터(완전히 다른 내용) 5점까지(동일한 내용) 점수를 매겨)

세번째 : 얼마나 유창한지, 일관성이 있는지 확인(문장 S2가 얼마나 자연스러운지 1점부터(전혀 자연스럽지 못함) 5점까지(아주 자연스러움) 점수를 매겨)

밑의 형식대로 생성해야해.

<스타일 변환 정확성>
문장 S1:{sentence}
문장 S2:{answer}
결과 = {score1}


<문장 내용 유지 평가>
문장 S1:{sentence} 
문장 S2:{answer}
결과 = {score2}


<유창성 평가>
문장 S2:{answer}
결과 = {score3}
"""

samples = pd.read_csv('./test_result.csv', sep=',')

for i in range(len(samples)):
    sentence = samples['original'][i]
    answer = samples['generate'][i]

    example_prompt = PromptTemplate(
        input_variables=["sentence", "answer", "score1", "score2", "score3"],
        template=template)

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="sentence: {sentence}\nanswer: {answer}",
        input_variables=["sentence", "answer"]
    )

    print(davinci3(
        prompt.format(sentence=sentence, answer=answer)
    ))
    print('=====================================================')