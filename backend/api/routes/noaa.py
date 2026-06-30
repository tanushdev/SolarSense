from fastapi import APIRouter
from backend.services.noaa_service import get_noaa

router = APIRouter()


@router.get("/noaa")
def get_noaa_status():
    return get_noaa().get_status()
