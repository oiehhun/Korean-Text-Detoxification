from classifier import Classifier, SHAP
from llm import Generator
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device('cpu')
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
generator = Generator()
model.load_state_dict(torch.load('./KcELECTRA_5_pth/checkpoint-4500/pytorch_model.bin',map_location='cpu'), strict=False)
model.to(device)
cls = Classifier(tokenizer,model,device)
shap = SHAP(tokenizer,model)
print('-'*100)
st.title("댓글 순화[cool] :sunglasses:")
# 댓글 입력
sentence = st.text_area("댓글을 입력해주세요: ")

need_convert = False
if sentence:
    cls_result = cls.sentence_predict(sentence)
    if cls_result and '악성댓글' in cls_result:
        need_convert = True
        print(cls_result)
        st.write(cls_result)
    else:
        st.write('정상 댓글입니다. 다른 댓글을 입력해보세요.')
        print(cls_result)
        need_convert = False



if need_convert:
    ori_sentence, masking_setence = shap.masking(sentence)
    st.write(f'ori: {ori_sentence}')
    st.write(f'mask : {masking_setence}')

    if st.button('댓글 순화 요청하기'):
        with st.spinner('댓글 순화 요청하기'):
            converted_sentence = generator.covert_sentence(ori_sentence,masking_setence)
            st.write(f'기존 댓글 : {sentence}')
            st.write(f'{converted_sentence}')