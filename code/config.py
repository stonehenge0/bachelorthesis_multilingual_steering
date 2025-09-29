import os
from dataclasses import dataclass


@dataclass
class Config:
    model_path: str
    model_alias: str

    steering_vector_path: str
    steering_layer: int
    steering_strengths: list[float]

    debug: bool = False
    device: str = "cuda:0"

    #  outpath for pipe results in: /runs/{model_alias}
    def artifact_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),  # one parent up
            "runs",
            f"{self.model_alias}",
        )