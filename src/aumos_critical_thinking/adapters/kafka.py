"""Kafka event publisher adapter for the AumOS Critical Thinking service.

Wraps aumos_common.events.EventPublisher with service-specific topic routing.
"""

from aumos_common.config import KafkaSettings
from aumos_common.events import EventPublisher


class CriticalThinkingEventPublisher(EventPublisher):
    """Kafka event publisher for critical thinking domain events.

    Publishes to the CRITICAL_THINKING topic for bias detections,
    atrophy interventions, and training lifecycle events.

    Inherits start/stop/publish lifecycle from aumos_common.events.EventPublisher.
    """

    def __init__(self, kafka_settings: KafkaSettings) -> None:
        """Initialise with Kafka connection settings.

        Args:
            kafka_settings: Kafka broker configuration from AumOS settings.
        """
        super().__init__(kafka_settings)
