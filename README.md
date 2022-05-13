# Industrial_systems_Engineering_PJ_Cloud
 
 
 # 🛫 Result

<img width="300" alt="스크린샷 2022-03-02 오전 4 20 39" src="https://user-images.githubusercontent.com/73048180/156298444-7cdd16fc-f36a-43fb-83b8-4acae916c732.png">


판례 임베딩 벡터 K-means 클러스터링

<img width="300" alt="스크린샷 2022-03-02 오전 4 21 00" src="https://user-images.githubusercontent.com/73048180/156298482-50a406e7-0cc3-4d0f-b3bc-af31dd5861a4.png">

부동산 판례 클러스터 Wordcloud

<img width="600" alt="스크린샷 2022-03-02 오전 4 21 30" src="https://user-images.githubusercontent.com/73048180/156298599-317f1d85-6489-4fff-a4fd-0cc85ed18090.png">

Input data 및 유사판례 간 네트워크 시각화

- 현재 처한 상황을 “줄 글"의 형태로 입력
- 현재 상황과 가장 유사한 판례 상위 10개 제공
- 현재 상황과 관련이 있는 참조 가능한 법 조항 제공
- 현재 상황에 대한 법원 기각여부 예측
- 유사도를 가중치로 한 판례 네트워크 시각화

---

# 🔍 기술 스택

- NLP(Natural Language Processing)
- Crawling
- Network Analysis

---

# ✍️  서론

- 프로젝트 배경
    - 변호사 선임 비용의 부담으로 나홀로 소송이 늘어가는 추세이다.
    - 나홀로 소송자는 합리적 증거를 제시하지 못하는 경우가 많다.
    - 나홀로 소송자는 법률지식이 부족하여 당사자의 권익을 구현하지 못하는 경우가 대다수 이다.
- 문제정의
    - 부동산 분쟁은 민사 소송중 30%를 차지할 정도로 빈도수가 높다.
    - 부동산 관련 법률 정보 서비스 제공이 필요하다.
    - 법은 일반인들에게 진입장벽이 높은 분야이다.
    - 법률용어가 아닌 일반용어로 구성된 문장으로 법률 정보 검색이 가능한 시스템이 필요하다.
- 프로젝트 목표
    - 법률 취약계층인 일반인을 대상으로 사건 내용과 관련된 판례, 법 조항, 기각여부 예측을 통한 법률 취약 계층의 재판 정보 비대칭 문제 완화

---

# 👨🏻‍💻담당 업무

### 프로젝트의 전반적인 개발을 담당

- Data 크롤링
    - 국가법률 정보센터 API를 활용하여 XML 형태의 하급심 판례 크롤링
    - LAWnB 동국대학교 계정을 이용하여 지방법원 판례 동적 크롤링
- 데이터 전처리
    - 판시사항이 없는 경우 Drop
    - 불용어 리스트 구축(723개) → 불 용어 제거 및 명사 추출
- 판례 Data 벡터화
    - TF - IDF Vectorizer로 임베딩 벡터 생성
    - t-sne 를 활용한 차원축소
- 유사도 분석
    - 판례 Data간 Cosine Similarity 값 계산 → 인접 행렬 생성
    - Cosine Similarity 기준 상위 10개 판례 추출
- 일반용어 맵핑
    - 일반용어를 법률용어로 Mapping 하기 위한 DB 구축
- 클러스터링
    - 판례별 유형 구분을 위한 K-Means 클러스터링 진행
- 네트워크 시각화
    - 상위 유사 판례 10개 + Input Data 의 인접행렬 생성
    - 생성한 인접행렬 기반 노드와 간선이 존재하는 네트워크 시각화

# 💡 느낀 점

- 비 정형 데이터를 활용한 비지도 학습을 이해함에 따라 비정형 데이터 분석 역량이 강화 되었다.
- 법률 정보가 필요한 일반인들에게 정보 전달의 시간 / 비용적 절감을 기대할 수있다.
- 개발 및 데이터 분석 측면에서 유의미한 결과가 도출 되었지만 사용자가 측면에서의 활용성이 부족해 사용자를 위한 서비스로 더 발전시켜 나갈 예정이다.

# 🛠 개발 내용

### 1. 데이터 수집

- ‘국가법령정보센터' API 권한 획득

![image](https://user-images.githubusercontent.com/73048180/156298720-0874f461-9ee1-4f7b-849d-bb98b6b95ec3.png)

### 2. 데이터 전처리 및 구조화

- 부동산 관련 판시사항 667개 추출(Nan 값 제거)
    
<img width="1000" alt="스크린샷 2022-03-02 오전 5 45 34" src="https://user-images.githubusercontent.com/73048180/156298791-de1ee056-d947-4465-9796-63f4ac6670d3.png">

    
- 불용어 리스트 구축
    
<img width="180" alt="스크린샷 2022-03-02 오전 5 45 12" src="https://user-images.githubusercontent.com/73048180/156298816-00c57155-e5de-4dc2-aa8e-6a8cad83691e.png">
    

### 3. 비 정형 데이터 벡터화

- 판례 데이터 토큰화 및 TF - IDF Vectorizer 적용

```python
def tokenizer_2(raw_texts, pos=["Noun","Alpha","Verb","Number"], stop_words=list(stop_words_df.get("불용어"))):
    nouns = []
  
    for noun in tagger.nouns(raw_texts):
        if noun not in stop_words and len(noun)>1:
            nouns.append(noun)
    return nouns

vectorize = TfidfVectorizer(
    tokenizer = tokenizer_2, # 문장에 대한 tokenizer (위에 정의한 함수 이용)
    min_df = 10,            # 단어가 출현하는 최소 문서의 개수
    sublinear_tf = True,    # tf값에 1+log(tf)를 적용하여 tf값이 무한정 커지는 것을 막음
    stop_words = list(stop_words_df.get("불용어"))
)

X = vectorize.fit_transform(posts)
pd.DataFrame(X.toarray())
```

- 2207 차원의 부동산 관련 판례 Vector 생성

### 4. 벡터간 유사도 계산

- Cosine Similarity 값 기준 내림차순 으로 유사 판례 10개 정렬

### 5. 차원축소

- PCA
    - 정보의 대부분을 유지하면서 차원을 축소하는 방법
    - 선형방식의 정사영 → 직관적인 해석이 어려움
- t-SNE
    - t 분포를 활용하여 고차원 데이터간 거리를 최대한 보존하며 저차원에서 학습하는 방법

**⇒ 고차원 데이터에 적합한 t-SNE 채택**

### 6. 일반용어 → 법률용어 맵핑

- Wordcloud 내 빈도 수 높은 단어를 기준으로 단어 Mapping
    
<img width="413" alt="스크린샷 2022-03-02 오전 5 57 59" src="https://user-images.githubusercontent.com/73048180/156298892-d924d6cc-92a8-4d1b-8c86-9d8e1e57068d.png">
    

### 7. 판례 클러스터링

- K-Means 클러스터링
    
<img width="838" alt="스크린샷 2022-03-02 오전 6 00 51" src="https://user-images.githubusercontent.com/73048180/156298904-4b270242-2853-4265-8921-80ebba0138c7.png">)
    

### 8. 네트워크시각화

<img width="1281" alt="스크린샷 2022-03-02 오전 6 01 23" src="https://user-images.githubusercontent.com/73048180/156298914-96039bfd-e0a7-4714-9675-4fd0a681323c.png">

    
