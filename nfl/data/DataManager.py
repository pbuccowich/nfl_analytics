from pathlib import Path
import pandas as pd

bool_columns = ["qb_kneel",
                "qb_spike",
                "aborted_play",
                "qb_dropback",
                "no_huddle",
                "shotgun",
                "timeout",
                "quarter_end",
                "sp",
                "goal_to_go",
                "special",
                "play",
                "out_of_bounds"
                "home_opening_kickoff",
                "success", 
                "aborted_play", 
                "div_game",
                "drive_inside_20",
                "play_deleted"
            ]

drop_columns = ["play_id",
                "jersey_number",
                "player_id,"
                "old_game_id",
                "player_name"]

class DataManager():
    @classmethod
    def get_data(cls, path_to_json: Path) -> pd.DataFrame:
        all_df = pd.DataFrame()
        path = Path(path_to_json)
        for f in path.glob('*.csv'):
            print(f"reading {f}")
            _df = pd.read_csv(f, 
                            low_memory=False,
                            parse_dates=['game_date'])
            _df['year'] = f.stem.split('_')[-1]
            all_df = pd.concat([all_df, _df], ignore_index=True)
        # Call via cls instead of self
        cleaned_data = cls.clean_data(all_df)
        return cleaned_data
    
    @classmethod
    def _normalize_to_boolean(cls, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        target_cols = [c for c in columns if c in df.columns]
        for col in target_cols:
            df[col] = df[col].astype('boolean')
        return df

    @classmethod
    def _drop_columns(cls, df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
        matched_columns = [
            col for col in df.columns 
            if any(sub in col for sub in cols_to_drop)
        ]
        
        print(matched_columns)
        cleaned_df = df.drop(columns=matched_columns)
        return cleaned_df

    @classmethod
    def clean_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        # Call via cls instead of self
        corrected_bool = cls._normalize_to_boolean(data, bool_columns)
        dropped_columns = cls._drop_columns(corrected_bool, drop_columns)
        return dropped_columns