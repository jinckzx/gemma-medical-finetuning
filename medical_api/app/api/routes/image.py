import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.model_registry import registry
from app.schemas.image import ImageDiagnosisResponse
from app.services import llm

router = APIRouter(prefix="/classify-image", tags=["Image Classification"])


@router.post(
    "",
    response_model=ImageDiagnosisResponse,
    summary="Classify skin-disease image and return MedGemma treatment advice",
)
async def classify_image(
    image: UploadFile = File(..., description="JPEG/PNG skin-lesion photo"),
    pulse_rate: str = Form(...),
    oxygen_level: str = Form(...),
    temperature: str = Form(...),
):
    """
    Upload an image of the skin lesion together with the patient's vitals.
    Returns the predicted disease class and MedGemma-generated treatment advice.

    If model confidence is below the threshold, no diagnosis is returned.
    """
    image_bytes = await image.read()

    # 1. decode the upload
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(422, f"Image processing failed: {e}")

    # 2. classify (blocking TensorFlow call, so keep it off the event loop)
    try:
        result = await run_in_threadpool(registry.predict_skin, pil_image)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Skin classifier is not available: {e}")
    except Exception as e:
        raise HTTPException(500, f"Image classification failed: {e}")

    disease, confidence = result["label"], result["confidence"]

    if confidence < settings.confidence_threshold:
        return ImageDiagnosisResponse(
            predicted_disease="No disease confidently detected",
            treatment_plan="",
        )

    # 3. generate advice (the first call also loads the text model, so run it in a thread too)
    def _generate():
        return llm.diagnose_from_disease(
            registry.hf_pipeline,
            disease=disease,
            pulse_rate=pulse_rate,
            oxygen_level=oxygen_level,
            temperature=temperature,
        )

    advice = await run_in_threadpool(_generate)

    return ImageDiagnosisResponse(
        predicted_disease=disease,
        treatment_plan=advice["treatment_plan"],
    )