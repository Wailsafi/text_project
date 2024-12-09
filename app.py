from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from gliner import GLiNER
import os
from fastapi.staticfiles import StaticFiles
from main import predict  


# Load model during app startup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates folder
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def load_model():
    global model
    try:
        model = GLiNER.from_pretrained("gliner-community/gliner_medium-v2.5")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request, "entities": None, "text": None})


# Endpoint to handle text input and return the detected entities
@app.post("/extract-entities", response_class=HTMLResponse)
async def extract_entities(request: Request, text: str = Form(...)):
    """Extract named entities from the input text."""
    try:
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        
        entities = predict(model, text)

        # Highlight entities in the text
        highlighted_text = text
        for entity in entities:
            highlighted_text = highlighted_text.replace(entity['text'], 
                f'<span class="highlight {entity["label"]}">{entity["text"]}</span>')

        # Render the HTML page with the highlighted text
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "entities": entities or [], "text": highlighted_text},
        )
    except HTTPException as he:
        raise he  # Re-raise known exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")
