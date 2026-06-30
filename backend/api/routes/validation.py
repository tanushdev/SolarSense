from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/validation", tags=["validation"])


class ValidationRequest(BaseModel):
    start_date: str = "2024-07-01"
    end_date: str = "2024-12-31"
    tolerance_minutes: int = 15


@router.get("/report")
def get_validation_report(start_date: str = "2024-07-01", end_date: str = "2024-12-31"):
    from backend.validation.goes_validator import compute_verification_report
    report = compute_verification_report(start_date=start_date, end_date=end_date)
    return report


@router.post("/report")
def post_validation_report(req: ValidationRequest):
    from backend.validation.goes_validator import compute_verification_report
    report = compute_verification_report(
        start_date=req.start_date,
        end_date=req.end_date,
    )
    return report


@router.get("/compare")
def compare_event(our_peak: str):
    from backend.validation.goes_validator import load_our_catalog, get_goes_catalog, match_events
    our = load_our_catalog()
    goes = get_goes_catalog()
    our_filtered = our[our["peak_time"] == our_peak] if "peak_time" in our.columns else our
    if our_filtered.empty:
        our_filtered = our[our["peak_time"].astype(str).str.contains(our_peak[:10])]
    result = match_events(our_filtered, goes)
    return result
