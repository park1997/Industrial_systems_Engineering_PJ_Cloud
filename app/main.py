from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
# from konlpy.tag import Okt
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import cm
import networkx as nx
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()
# 미들웨어
app.mount(str(BASE_DIR/"static"), StaticFiles(directory=BASE_DIR/"static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR/"templates"))



info_df = pd.read_excel(BASE_DIR/"data/budongsan.xlsx",engine='openpyxl')
info2_df = pd.read_excel(BASE_DIR/"data/estate2.xlsx",engine='openpyxl')
df_mapping = pd.read_excel(BASE_DIR/"data/wordmapping.xlsx",engine='openpyxl')
df_mapping = df_mapping.dropna()
stop_words_df = pd.read_excel(BASE_DIR/"data/stopwords.xlsx", engine='openpyxl')


def tokenizer(raw_texts, pos=["Noun","Alpha","Verb","Number"], stop_words=list(stop_words_df.get("불용어"))):
    nouns = []
    tagger = Mecab()
    for noun in tagger.nouns(raw_texts):
        if noun not in stop_words and len(noun)>1:
            nouns.append(noun)
    return nouns



@app.get('/') 
async def root(request : Request): 
	return templates.TemplateResponse("index.html",{"request" : request,
                                                    "subject" : "나홀로 소송을 준비하는 일반인을 위한 법률 정보 검색 시스템",
                                                    "title":"빈센조"})

@app.get('/search') 
async def search(request : Request, query : str):
    global oo, info_df, info2_df, df_mapping
    print(query)

    # 전처리, 일반용어 맵핑
    input_data = query
    law_to_original = []
    for idx,i in enumerate(df_mapping["일반용어"]):
        temp = i.split(",")
        for j in temp:
            if len(j.strip()) ==0:
                continue
            if j.strip() in input_data:
                input_data = input_data.replace(j.strip(),df_mapping["부동산관련 법률용어"].iloc[idx].split("(")[0])
                law_to_original.append([j.strip(),df_mapping["부동산관련 법률용어"].iloc[idx].split("(")[0]])

    info_df = pd.concat([info2_df,info_df])
    info_df.reset_index()
    input_df = pd.Series([0,"","",input_data],index = ["ID","참조조문","주문","이유"])
    info_df = info_df.append(input_df,ignore_index= True)
    info_df.drop_duplicates(['ID'])
    posts = info_df.get("이유")



    vectorize = TfidfVectorizer(
        tokenizer = tokenizer, # 문장에 대한 tokenizer (위에 정의한 함수 이용)
        min_df = 10,            # 단어가 출현하는 최소 문서의 개수
        sublinear_tf = True,    # tf값에 1+log(tf)를 적용하여 tf값이 무한정 커지는 것을 막음
        stop_words = list(stop_words_df.get("불용어"))
    )

    X = vectorize.fit_transform(posts)
    cosine_result = {}
    error_reslut = {}
    target_id = 0
    x = info_df[info_df['ID'] == target_id].index[0]  # target_id의 Id값의 index

    for y in range(len(info_df)):
        try:
            cosine_result[info_df['ID'].iloc[y]] = cosine_similarity(X[x], X[y])[0][0]
        except:
            error_reslut[info_df["ID"].iloc[y]] = cosine_similarity(X[x], X[y])
    sorted_cosine_dic = sorted(cosine_result.items(), key = lambda x:x[1], reverse = True)
    p_url = []
    for id_ in sorted_cosine_dic[0:11]:
        if len(str(id_[0])) < 5:
            continue
        url = "https://www.law.go.kr/precInfoP.do?mode=0&precSeq={}&vSct=*".format(id_[0])
        idx_ = info_df[info_df['ID'] == id_[0]].index[0]
        p_url.append([info_df.iloc[idx_]["이유"][:30]+"...",url])
    
    # 관련 법
    related_law = []
    for i in sorted_cosine_dic[0:11]:
        temp = ""
        for j in str(info_df[info_df["ID"]==i[0]]["참조조문"].iloc[0]).split(","):
            if j.strip() =="nan":
                continue
            temp += j.strip()+" "
        temp = temp.strip()
        if len(temp) != 0:
            related_law.append(temp.strip())
    
    # 기각 여부
    cnt_기각 = 0
    cnt_승소 = 0
    cnt_일부승소 = 0
    verdict_list = []
    for i in sorted_cosine_dic[1:11]:
        # info_df[info_df["ID"]==i[0]]["주문"].iloc[0]
        if '나머지 청구를 기각한다' in info_df[info_df["ID"]==i[0]]["주문"].iloc[0] or "나머지 항소를 기각한다" in info_df[info_df["ID"]==i[0]]["주문"].iloc[0] or "나머지 청구" in info_df[info_df["ID"]==i[0]]["주문"].iloc[0] or "지급하라" in info_df[info_df["ID"]==i[0]]["주문"].iloc[0]:
            cnt_일부승소 += 1
            verdict_list.append([info_df[info_df["ID"]==i[0]]["주문"].iloc[0], "결과 : 일부승소"])
        elif "기각" in info_df[info_df["ID"]==i[0]]["주문"].iloc[0]:
            cnt_기각 += 1
            verdict_list.append([info_df[info_df["ID"]==i[0]]["주문"].iloc[0], "결과 : 기각"])
        else:
            cnt_승소 += 1
            verdict_list.append([info_df[info_df["ID"]==i[0]]["주문"].iloc[0], "결과 : 승소"])
        
    verdict_result = "기각 :{}건;승소 : {}건;일부 승소 {}건".format(cnt_기각,cnt_승소,cnt_일부승소)
    verdict_result = verdict_result.split(";")


    target_cosine_id = sorted_cosine_dic[0:11]
    cosine_result = []
    for target_id,cos_sim in target_cosine_id:
        tmp = []
        x = info_df[info_df['ID'] == target_id].index[0]
        for target_id2,cos_sim2 in target_cosine_id:
            y = info_df[info_df['ID'] == target_id2].index[0]
            tmp.append(cosine_similarity(X[x], X[y])[0][0])
        cosine_result.append(tmp)
    cosine_result_df = pd.DataFrame(cosine_result)

    result_index = []
    for i in target_cosine_id:
        result_index.append(info_df[info_df["ID"] == i[0]].index[0])
    result_id = []
    for i in sorted_cosine_dic[0:11]:
        if int(i[0]) == 0:
            result_id.append("INPUT")
            continue
        result_id.append(int(i[0]))
    network = nx.from_numpy_matrix(cosine_result_df.to_numpy())
    G=nx.Graph(name='Law Interaction Graph')
    for i in range(len(cosine_result_df)):
        for j in range(i+1,len(cosine_result_df)):
            a = result_id[i] # cos_sim a node
            b = result_id[j] # cos_sim b node
            w = float(cosine_result_df.iloc[i].iloc[j]) # score as weighted edge where high scores = low weight
            G.add_weighted_edges_from([(a,b,w)])

    def rescale(l,newmin,newmax):
        arr = list(l)
        return [(x-min(arr))/((max(arr)-min(arr))+0.1)*(newmax-newmin)+newmin for x in arr]
    # use the matplotlib plasma colormap
    graph_colormap = cm.get_cmap('plasma', 12)
    # node color varies with Degree
    c = rescale([G.degree(v) for v in G],0.0,0.9) 
    c = [graph_colormap(i) for i in c]
    # node size varies with betweeness centrality - map to range [10,100] 
    bc = nx.betweenness_centrality(G) # betweeness centrality
    s =  rescale([v for v in bc.values()],5000,7000)
    # edge width shows 1-weight to convert cost back to strength of interaction 
    ew = rescale([float(G[u][v]['weight']) for u,v in G.edges],0.5,20)
    # edge color also shows weight
    ec = rescale([float(G[u][v]['weight']) for u,v in G.edges],0.1,1)
    ec = [graph_colormap(i) for i in ec]
    pos = nx.spring_layout(G)
    plt.figure(figsize=(20,20),facecolor=[0.7,0.7,0.7,0.4])
    nx.draw_networkx(G, pos=pos, with_labels=True, node_color=c, node_size=s,edge_color= ec,width=ew,
                    font_color='white',font_weight='bold',font_size='18')
    plt.axis('off')
    # plt.show()
    plt.savefig(BASE_DIR/"static/network_img.jpg")



    return templates.TemplateResponse("index.html",{"request" : request,
                                                    "subject" : "나홀로 소송을 준비하는 일반인을 위한 법률 정보 검색 시스템",
                                                    "title": "빈센조",
                                                    "p_url": p_url,
                                                    "related_law": related_law,
                                                    "verdict_list": verdict_list,
                                                    "verdict_result": verdict_result,
                                                    "law_to_original": law_to_original,
                                                    "input_data": input_data,
                                                    "query": query,
                                                    })



@app.get("/items/{id}", response_class=HTMLResponse) 
async def read_item(request: Request, id: str): 
	return templates.TemplateResponse("item.html", {"request": request, "id":id, "data":"HELLO FASTAPI"}) 

@app.get("/test")
async def main():
    return FileResponse("app/static/network_img.jpg")
