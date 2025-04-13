from fastapi import FastAPI, File, UploadFile, HTTPException
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import shutil, os

app = FastAPI()

# Set via Render environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("HF_REPO_ID")  # e.g. "your-username/your-dataset"
UPLOAD_PATH = "images"

hf_api = HfApi()

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        remote_path = f"{UPLOAD_PATH}/{file.filename}"

        hf_api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=remote_path,
            repo_id=REPO_ID,
            repo_type="dataset",  # or "space", "model"
            token=HF_TOKEN,
        )

        os.remove(temp_path)

        file_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{remote_path}"
        return {"success": True, "file_url": file_url}

    except HfHubHTTPError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
