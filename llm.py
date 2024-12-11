from dotenv import load_dotenv
load_dotenv()
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate


class Generator:
    def __init__(self):
        self.llm = OpenAI()
        self.davinci3 = OpenAI(
            model_name = 'text-davinci-003',
            max_tokens = 1000,
            temperature = 0
        )
        self.examples = [
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
        self.template = """너는 문장 스타일 변환을 하는 역할을 할거야. 원래 댓글과, 해당 댓글이 악성 댓글로 판단되는데 중요한 영향을 미친(=feature importance가 높은) 단어를 masking 처리한 댓글이 주어지면 기존 댓글의 context는 유지하면서 주어진 스타일에 맞는 댓글을 생성하는 것이 너의 임무야.
            sentence : {sentence} 
            masking_sentence : {masking_sentence}
            순화 댓글: {answer}
            """
        self.example_prompt = PromptTemplate(
            input_variables=["sentence", "masking_sentence", "answer"], 
            template=self.template)

        self.prompt =  FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=self.example_prompt,
            suffix="sentence: {origin}\nmasking_sentence: {masked}",
            input_variables=["origin", "masked"]
        )
        

    def covert_sentence(self,ori_setence,mask_sentence):
        
        result = self.davinci3(self.prompt.format(origin = ori_setence, masked = mask_sentence))
        print(result)

        return result