# init model
from util.ner_predict2 import keyword_predict
from util.intent_predict import intent0_parameter,intent0_model,intent1_parameter,intent1_model,predict
from util.neo4j_model import GraphSearch

#service
from flask_cors import cross_origin
from flask import Flask,request,redirect,url_for
import requests,json

from mychatbot import template,CorpusSearch,InterNet
from operator import itemgetter

intent0_tag = {
    '0':'基于知识图谱的问答',
    '1':'基于机器人个人属性的问答',
    '2':'基于语料的问答'
}

intent1_tag = {
    '0':'询问科目有哪些课程',
    '1':'询问科目有哪些知识点',
    '2':'询问科目有哪些例题',
    # '3':'询问课程是什么学科的',
    '4':'询问课程有哪些知识点',
    '5':'询问课程有哪些例题',
    '6':'询问是哪个学科的知识点',
    '7':'询问是哪个课程的知识点',
    '8':'询问这个知识点有哪些例题需要掌握',
    '9':'询问作者有哪些诗句',
    '10':'询问作者个人简介',
    '11':'询问诗是谁写的',
    '12':'询问诗的诗句',
    '13':'询问诗的翻译',
    '14':'询问诗的类型',
    # '15':'询问诗来自那篇课文',
    '16':'询问这种类型的古诗有哪些',
    # '17':'询问这个年级学过哪些诗',
}

# global 
app = Flask(__name__)
# init the chatbot
template_model = template()
CorpusSearch_model = CorpusSearch()
InterNet_model = InterNet()
GraphSearch_model = GraphSearch()

def takelong(ins):
    return len(ins[0])

def rebuildiins(ins,entity_list):
    new_ins = {}
    left_ind = set(range(len(ins)))
    for i in entity_list:
        left_ind -= set(range(i[-1][0],i[-1][-1]+1))
        new_ins[i[-1][0]] = i[1]
    for i in left_ind:
        new_ins[i] = ins[i]
    new_id = list(new_ins.keys())
    new_id.sort()
    return itemgetter(*new_id)(new_ins)
    

@app.route('/test', methods=['GET', 'POST'])
@cross_origin()
def myfirst_service():
    if request.method ==  "POST":
        #sta_post = time.time()
        data = request.get_data().decode()
        data = json.loads(data)
        question = data['question']

        '''
        step 1：首先进行实体识别，识别到的实体，进行替换
        step 2：进行第一次意图识别
        step 3：根据第一次意图识别的结果，选择性调用接口；若识别结果为基于知识图谱的方式进行问答；则进行下一步的意图识别
        step 4：进行第二次意图识别，并根据识别的结果，根据预先设定的搜索方式，查询答案
        '''
        # 进行实体识别
        entity_list = keyword_predict(question)
        print('实体识别：',entity_list)
        # 按照实体名大的进行排序
        entity_list.sort(key = takelong)
        entity_list = entity_list[::-1]
        # 根据实体识别的结果重建输入
        new_question = rebuildiins(question,entity_list)
        print('基于ner重建后的提问：',new_question)
        # 进行第一次意图识别
        intent0 = predict([list(new_question)],intent0_model,intent0_parameter)[0]
        print('意图识别0模型识别结果：',intent0)
        intent1 = None
        answer = None
        if intent0 == 0:
            # 进行第二次意图识别
            intent1 = predict([list(new_question)],intent1_model,intent1_parameter)[0]
            if len(entity_list) == 1:
                print('意图识别1模型识别结果：',intent1,entity_list[0])
                try:
                    answer = eval('GraphSearch_model.forintent'+str(intent1)+'(entity_list[0])')
                except:
                    answer = None
        if intent0 == 1:
            answer = template_model.search_answer(question)
        if intent0 == 2:
            answer = CorpusSearch_model.search_answer(question)
        if answer is None:
            # try:
            #     answer = InterNet_model.search_answer(question)
            # except:
                answer = '对不起啊，小智无法解决这个问题'
        return json.dumps(answer,ensure_ascii=False)


        
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8081,threaded=True)
