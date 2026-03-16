from abc import ABC, abstractmethod
from typing import Optional


class BaseDataLoader(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self) -> list[dict]:
        pass


class MSMARCOLoader(BaseDataLoader):
    def __init__(self, data_path: Optional[str] = None):
        super().__init__("msmarco")
        self.data_path = data_path

    def load(self) -> list[dict]:
        return []


class HotpotQALoader(BaseDataLoader):
    def __init__(self, data_path: Optional[str] = None):
        super().__init__("hotpotqa")
        self.data_path = data_path

    def load(self) -> list[dict]:
        return []


class BEIRLoader(BaseDataLoader):
    def __init__(self, data_path: Optional[str] = None):
        super().__init__("beir")
        self.data_path = data_path

    def load(self) -> list[dict]:
        return []
