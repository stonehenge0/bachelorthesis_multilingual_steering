import os
from dataclasses import dataclass


@dataclass
class Config:
    model_path: str
    model_alias: str

    steer_type: str
    steering_folder: str
    steering_vector_path: str
    steering_layer: int
    steering_token_position: int
    steering_strengths: list[float]

    debug: bool = False
    device: str = "cuda:0"

    # set clean outpath for pipe results in: pipeline/runs/{model_alias}
    def artifact_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "runs",
            f"{self.model_alias}",
        )
