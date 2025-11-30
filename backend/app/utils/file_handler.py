# app/utils/file_handler.py

from pathlib import Path
from fastapi import UploadFile
import aiofiles
import shutil


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """
    Save an uploaded file to disk asynchronously
    
    Args:
        upload_file: FastAPI UploadFile object
        destination: Path where to save the file
    """
    async with aiofiles.open(destination, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)


def cleanup_files(directory: Path, session_id: str) -> None:
    """
    Clean up files associated with a session
    
    Args:
        directory: Directory containing files
        session_id: Session ID to match files
    """
    for file_path in directory.glob(f"{session_id}*"):
        try:
            file_path.unlink()
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """
    Validate file size
    
    Args:
        file_size: Size of file in bytes
        max_size_mb: Maximum allowed size in MB
    
    Returns:
        True if valid, False otherwise
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()