import pandas as pd
import pandera as pa
import json

class DataCleaner:
    def __init__(self, schema_path: str):
        with open(schema_path, 'r') as f:
            config = json.load(f)
        
        # Convert string types back to proper data types for Pandera
        schema_dict = {
            col: pa.Column(dtype=self._map_dtype(dtype)) 
            for col, dtype in config['columns'].items()
        }
        
        self.schema = pa.DataFrameSchema(
            schema_dict, 
            coerce=config.get('coerce', True)
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