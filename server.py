# uvicorn은 ASGI웹 어플리케이션을 실행하기위한 도구
import uvicorn

if __name__=="__main__":
    uvicorn.run("app.main:app",host="localhost",port=8000,reload=True)
    # uvicorn.run("app.main:app",host="localhost",port=8000)
    # uvicorn.run("app.main:app",host="0.0.0.0",port=80,reload=False)



# source venv/bin/activate
# pip freeze > requirements.txt
# pip3 install mecab-python3