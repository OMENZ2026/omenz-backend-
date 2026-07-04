from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="OMENZ Backend", version="0.1.0")

@app.get("/")
def root():
    return {
        "service": "OMENZ Backend",
        "status": "online",
        "message": "OMENZ Stack backend is running."
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/auth/patreon/callback")
async def patreon_callback(request: Request):
    return JSONResponse({
        "status": "received",
        "query_params": dict(request.query_params)
    })
