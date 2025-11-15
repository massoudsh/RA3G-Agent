"""Agent status tracking module for monitoring health and performance metrics."""
import time
from typing import Dict, Optional
from datetime import datetime
from collections import deque
import threading

class AgentStatusTracker:
    """Tracks metrics for a single agent."""
    
    def __init__(self, agent_name: str, slow_threshold_ms: int = 5000):
        self.agent_name = agent_name
        self.slow_threshold_ms = slow_threshold_ms
        self.start_time = time.time()
        self.last_activity: Optional[float] = None
        self.error_count = 0
        self.request_count = 0
        self.latency_history = deque(maxlen=10)  # Keep last 10 latencies
        self.lock = threading.Lock()
    
    def record_activity(self, latency_ms: float = None, error: bool = False):
        """Record an activity (success or error)."""
        with self.lock:
            self.last_activity = time.time()
            self.request_count += 1
            if error:
                self.error_count += 1
            if latency_ms is not None:
                self.latency_history.append(latency_ms)
    
    def get_status(self) -> Dict:
        """Get current status summary."""
        with self.lock:
            uptime_seconds = time.time() - self.start_time
            avg_latency = (
                sum(self.latency_history) / len(self.latency_history)
                if self.latency_history else 0.0
            )
            
            # Determine status
            if self.error_count > 5 or (self.last_activity and (time.time() - self.last_activity) > 300):
                status = "down"
            elif avg_latency > self.slow_threshold_ms:
                status = "slow"
            else:
                status = "healthy"
            
            return {
                "agent": self.agent_name,
                "status": status,
                "uptime_seconds": round(uptime_seconds, 2),
                "last_activity": (
                    datetime.fromtimestamp(self.last_activity).isoformat()
                    if self.last_activity else None
                ),
                "latency_ms": round(avg_latency, 2),
                "error_count": self.error_count,
                "request_count": self.request_count,
            }
    
    def get_detailed_status(self) -> Dict:
        """Get detailed status with additional metrics."""
        base = self.get_status()
        with self.lock:
            base.update({
                "recent_latencies": list(self.latency_history),
                "slow_threshold_ms": self.slow_threshold_ms,
            })
        return base


class StatusRegistry:
    """Registry for all agent status trackers."""
    
    def __init__(self):
        self.trackers: Dict[str, AgentStatusTracker] = {}
        self.lock = threading.Lock()
    
    def get_tracker(self, agent_name: str) -> AgentStatusTracker:
        """Get or create a tracker for an agent."""
        with self.lock:
            if agent_name not in self.trackers:
                self.trackers[agent_name] = AgentStatusTracker(agent_name)
            return self.trackers[agent_name]
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status for all agents."""
        with self.lock:
            return {name: tracker.get_status() for name, tracker in self.trackers.items()}
    
    def get_agent_status(self, agent_name: str) -> Optional[Dict]:
        """Get detailed status for a specific agent."""
        with self.lock:
            if agent_name in self.trackers:
                return self.trackers[agent_name].get_detailed_status()
            return None


# Global registry instance
status_registry = StatusRegistry()

