from fastapi import FastAPI


app = FastAPI(title="AI Retail Inventory Optimizer API")


@app.get("/")
def read_root():
    return {"message": "Backend is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
