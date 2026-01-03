from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

class ReserveProduct(BaseModel):
    name: str
    type: str  # "regulation", "spinning", "non_spinning"
    direction: str # "up", "down"
    duration_minutes: int # Energy backing requirement

class IsoAdapter(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def dispatch_interval_minutes(self) -> int:
        pass
    
    @property
    @abstractmethod
    def reserve_products(self) -> List[ReserveProduct]:
        pass

    @abstractmethod
    def get_market_data(self, start_time: str, end_time: str) -> Dict[str, Any]:
        """
        Placeholder for fetching or mocking market prices/data.
        """
        pass
