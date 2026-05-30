"""Alerting system — threshold-based failure alerts with cooldown."""

import logging
from datetime import datetime, timedelta

from src.config import get_config

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages failure alerts with cooldown to prevent alert fatigue."""

    def __init__(self):
        cfg = get_config()["monitoring"]
        self.threshold = cfg["failure_threshold"]
        self.cooldown = timedelta(seconds=cfg["alert_cooldown_seconds"])
        self._last_alert: dict[str, datetime] = {}
        self.alert_history: list[dict] = []
        self.total_alerts = 0

    def check_and_alert(self, equipment_id: str, probability: float) -> dict | None:
        """Check if alert should be triggered. Returns alert dict or None."""
        if probability < self.threshold:
            return None

        now = datetime.now()
        last = self._last_alert.get(equipment_id)

        if last and (now - last) < self.cooldown:
            return None  # Still in cooldown

        self._last_alert[equipment_id] = now
        self.total_alerts += 1

        alert = {
            "equipment_id": equipment_id,
            "probability": round(probability, 4),
            "severity": self._severity(probability),
            "timestamp": now.isoformat(),
            "message": f"Equipment {equipment_id}: failure risk {probability:.1%} "
                       f"({self._severity(probability).upper()})",
        }

        self.alert_history.append(alert)
        logger.warning("ALERT: %s", alert["message"])
        return alert

    @staticmethod
    def _severity(probability: float) -> str:
        if probability >= 0.9:
            return "critical"
        elif probability >= 0.8:
            return "high"
        elif probability >= 0.7:
            return "medium"
        return "low"

    def get_recent_alerts(self, n: int = 20) -> list[dict]:
        """Get the most recent n alerts."""
        return self.alert_history[-n:]

    def get_alert_summary(self) -> dict:
        """Summary of alerts by severity."""
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "total": self.total_alerts}
        for a in self.alert_history:
            summary[a["severity"]] += 1
        return summary
