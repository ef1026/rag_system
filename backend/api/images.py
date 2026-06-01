from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.constants import IMAGE_MEDIA_TYPES
from backend.core.ids import safe_document_id
from backend.core.paths import document_path
from backend.services.image_service import get_document_image_asset, image_file_exists

router = APIRouter()


@router.get("/api/documents/{document_id}/images/{image_id}")
async def get_document_image(document_id: str, image_id: str) -> FileResponse:
    normalized_document_id = safe_document_id(document_id)
    document_path(normalized_document_id)
    asset = get_document_image_asset(normalized_document_id, image_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    if not image_file_exists(asset):
        raise HTTPException(status_code=404, detail="Image file not found.")

    media_type = IMAGE_MEDIA_TYPES.get(asset.absolute_path.suffix.lower())
    if media_type is None:
        raise HTTPException(status_code=404, detail="Unsupported image type.")
    return FileResponse(
        asset.absolute_path,
        media_type=media_type,
        filename=asset.filename,
    )
