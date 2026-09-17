# -*- coding: utf-8 -*-
"""
Application Layer - Training Use Cases

Orchestrates domain objects to perform training operations.
Contains no business rules - delegates to domain entities and services.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional

from ..domain.training.entities import TrainingSession, TrainingStatus
from ..domain.training.value_objects import TrainingConfig
from ..domain.training.repositories import (
    ITrainingRepository,
    ITrainingSessionRepository,
    ITrainingRunner,
)
from ..domain.training.services import TrainingConfigValidator

logger = logging.getLogger(__name__)


class StopTrainingUseCase:
    """Stop a running training session."""

    def __init__(
        self,
        session_repo: ITrainingSessionRepository,
        runner: ITrainingRunner,
    ):
        self._session_repo = session_repo
        self._runner = runner

    def execute(self) -> Tuple[bool, str]:
        """
        Stop the current training session.

        Returns:
            Tuple of (success, message)
        """
        session = self._session_repo.get_current_session()
        if not session or not session.is_active:
            return False, "No active training to stop"

        try:
            session.stop()
            if session.process_id:
                self._runner.stop(session.process_id)
            self._session_repo.save_session(session)
            return True, "Training stop requested"
        except Exception as e:
            return False, f"Failed to stop training: {e}"


class GetTrainingStatusUseCase:
    """Get current training session status."""

    def __init__(self, session_repo: ITrainingSessionRepository):
        self._session_repo = session_repo

    def execute(self) -> Optional[Dict[str, Any]]:
        """
        Get current training status.

        Returns:
            Status dict or None if no session
        """
        session = self._session_repo.get_current_session()
        if not session:
            return None

        return {
            "id": session.id,
            "status": session.status.value,
            "epoch": session.current_epoch,
            "step": session.current_step,
            "total_steps": session.total_steps,
            "progress": session.progress_pct,
            "loss": session.last_loss,
            "loss_components": session.loss_components,
            "error": session.error_message,
        }
