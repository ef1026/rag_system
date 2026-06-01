from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import (
    CacheResponse,
    Folder,
    FolderCreate,
    FolderPatch,
    ManagedFile,
    ManagedFilePatch,
    ManagedFileRegister,
    RegisterManagedFileResponse,
    SyncExistingDocumentsResponse,
    TagSummary,
)
from backend.services.file_manager_service import (
    create_folder,
    delete_file,
    delete_folder,
    list_files,
    list_folders,
    list_tags,
    patch_file,
    patch_folder,
    register_file,
    sync_existing_documents,
)

router = APIRouter()


@router.get("/api/file-manager/files", response_model=list[ManagedFile])
async def read_managed_files(
    folder_id: str | None = None,
    tag: str | None = None,
    course: str | None = None,
    archived: bool | None = None,
    pinned: bool | None = None,
    q: str | None = None,
) -> list[ManagedFile]:
    return list_files(
        folder_id=folder_id,
        tag=tag,
        course=course,
        archived=archived,
        pinned=pinned,
        q=q,
    )


@router.post(
    "/api/file-manager/files/register",
    response_model=RegisterManagedFileResponse,
)
async def register_managed_file(
    payload: ManagedFileRegister,
) -> RegisterManagedFileResponse:
    return register_file(payload)


@router.patch("/api/file-manager/files/{file_id}", response_model=ManagedFile)
async def update_managed_file(
    file_id: str,
    payload: ManagedFilePatch,
) -> ManagedFile:
    return patch_file(file_id, payload)


@router.delete("/api/file-manager/files/{file_id}", response_model=CacheResponse)
async def remove_managed_file(file_id: str) -> CacheResponse:
    delete_file(file_id)
    return CacheResponse(ok=True)


@router.get("/api/file-manager/folders", response_model=list[Folder])
async def read_folders() -> list[Folder]:
    return list_folders()


@router.post("/api/file-manager/folders", response_model=Folder)
async def add_folder(payload: FolderCreate) -> Folder:
    return create_folder(payload)


@router.patch("/api/file-manager/folders/{folder_id}", response_model=Folder)
async def update_folder(folder_id: str, payload: FolderPatch) -> Folder:
    return patch_folder(folder_id, payload)


@router.delete("/api/file-manager/folders/{folder_id}", response_model=CacheResponse)
async def remove_folder(folder_id: str) -> CacheResponse:
    delete_folder(folder_id)
    return CacheResponse(ok=True)


@router.get("/api/file-manager/tags", response_model=list[TagSummary])
async def read_tags() -> list[TagSummary]:
    return list_tags()


@router.post(
    "/api/file-manager/sync-existing-documents",
    response_model=SyncExistingDocumentsResponse,
)
async def sync_documents() -> SyncExistingDocumentsResponse:
    return sync_existing_documents()
