import pandas as pd
import pandera.pandas as pa
import json

class DataCleaner:
    def __init__(self, schema_path: str):
        with open(schema_path, 'r') as f:
            config = json.load(f)
            
        definitions = config.get("definitions", {})
        schema_dict = {}
            
        for col, cfg in config['columns'].items():
            # If cfg is just a string, convert it to a dict for consistency
            if isinstance(cfg, str):
                cfg = {"type": cfg, "nullable": True}
                
            # Resolve checks (e.g., ref)
            checks = []
            if "ref" in cfg and cfg["ref"] in definitions:
                checks.append(pa.Check.isin(definitions[cfg["ref"]]))
                
            # Map the type string to the actual type object
            dtype = self._map_dtype(cfg.get("type", "object"))
                
            schema_dict[col] = pa.Column(
                dtype=dtype,
                nullable=cfg.get("nullable", True),
                checks=checks
            )
            
        self.schema = pa.DataFrameSchema(
            schema_dict, 
            coerce=True, 
            strict='filter'
        )

    @staticmethod
    def _map_dtype(dtype_str):
        # Map strings to numpy/pandas/python types
        mapping = {
            "int": int,
            "float": float,
            "bool": bool,
            "datetime64[ns]": "datetime64[ns]",
            "str": str
        }
        return mapping.get(dtype_str, object)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.schema.validate(df, lazy=True)