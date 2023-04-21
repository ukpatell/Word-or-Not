import pickle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()


templates = Jinja2Templates(directory="templates")

with open('output/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('output/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/api/check-word')
def checkWord(word:str):
    word_to_check = word
    test = vectorizer.transform([word_to_check])
    prediction = model.predict(test)
    if prediction[0] == 1:
        return {"result":f"{word_to_check} is an English word."}
    else:
        return {"result":f"{word_to_check} is not an English word."}


if __name__=='__main__':
    uvicorn.run(app,port=8080)