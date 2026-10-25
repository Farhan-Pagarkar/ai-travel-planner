from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


class HumanReviewInput(BaseModel):
    action: ReviewAction = Field(..., description="Action: approve, reject, or modify.")
    feedback: Optional[str] = Field(default=None, description="Feedback message, required when rejecting.")
    modifications: Optional[str] = Field(
        default=None,
        description="Specific modification instructions, required when requesting modifications.",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "HumanReviewInput":
        if self.action == ReviewAction.REJECT:
            if not self.feedback or not self.feedback.strip():
                raise ValueError("Non-empty feedback is required when rejecting an itinerary.")
        elif self.action == ReviewAction.MODIFY:
            instructions = (self.modifications or "").strip() or (self.feedback or "").strip()
            if not instructions:
                raise ValueError(
                    "Specific modification instructions are required when requesting modifications."
                )
            # Normalize to ensure modifications is populated if feedback was provided instead
            if not self.modifications:
                self.modifications = instructions
        return self
