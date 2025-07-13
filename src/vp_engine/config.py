import os
import json
from dataclasses import dataclass, field
from jsonschema import validate, ValidationError

FILEDIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    """Dataclass to hold the configuration for the perception engine.
    If new parameters are added, the schema must be updated accordingly."""

    config_path: str
    foundation_model: dict = field(init=False)
    model_heads: list[dict] = field(init=False)
    logging: dict = field(init=False)
    output_dir: str = field(init=False)
    canonical_image_shape_hwc: dict = field(init=False)
    queue_sizes: dict = field(init=False)
    schema_path: str = os.path.join(FILEDIR, "schemas", "vp_engine_config.json")

    def __post_init__(self):
        try:
            config = self._validate_config()
        except ValidationError as e:
            raise ValidationError(f"The configuration file is not valid: {e}")

        # Update the dataclass with the validated config
        self.__dict__.update(config)

    def _validate_config(self):
        # Load the schema
        with open(self.schema_path, "r") as f:
            schema = json.load(f)

        # Load the config
        with open(self.config_path, "r") as f:
            config = json.load(f)

        # Validate the config
        validate(instance=config, schema=schema)

        return config
