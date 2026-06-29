import os
import argparse
import subprocess
import pandas as pd
from dotenv import load_dotenv

# Try to import mosqlient and give a clear warning if it is not installed
try:
    from mosqlient import upload_prediction
except ImportError:
    print("\n[WARNING] 'mosqlient' library not found.")
    print("Please install it using: pip install -U mosqlient\n")
    upload_prediction = None

def get_git_commit():
    """Automatically retrieve the current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Submit IMDC predictions state-by-state to the Mosqlimate platform.")
    parser.add_argument("--file", required=True, help="Path to the combined submission CSV file.")
    parser.add_argument("--desc", required=True, help="Description for this forecast upload (e.g., 'Validation Round 2022-2023').")
    parser.add_argument("--commit", help="Git commit hash. Defaults to the current git HEAD commit.")
    parser.add_argument("--disease", default="A90", help="ICD-10 disease code: 'A90' for Dengue (default), 'A92.0' for Chikungunya.")
    parser.add_argument("--case-definition", default="probable", choices=["probable", "reported"], help="Case definition type.")
    parser.add_argument("--repository", default="graeme-dor/3rd_imdc_ceri_returnoftheforecast", help="Registered model repository name.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry-run local check without making API calls.")

    args = parser.parse_args()

    # Verify that the upload library is available
    if not args.dry_run and upload_prediction is None:
        raise ImportError("Cannot upload predictions: 'mosqlient' is not installed.")

    # Load environment variables (contains your API_KEY)
    load_dotenv()
    api_key = os.getenv("API_KEY")

    if not args.dry_run and not api_key:
        raise ValueError("API_KEY environment variable not found. Please define it in your .env file.")

    # Resolve commit hash
    commit_hash = args.commit if args.commit else get_git_commit()
    if not commit_hash:
        raise ValueError("Could not determine git commit. Please specify it manually using the --commit argument.")

    # Check if input file exists
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Forecast file not found: {args.file}")

    print(f"Loading forecasts from {args.file}...")
    pred_df = pd.read_csv(args.file)

    # Ensure necessary columns are present
    required_cols = ["uf", "date", "pred", "lower_50", "upper_50", "lower_80", "upper_80", "lower_90", "upper_90", "lower_95", "upper_95"]
    for col in required_cols:
        if col not in pred_df.columns:
            raise KeyError(f"Missing required column in forecast CSV: '{col}'")

    # Load geocode mappings from data/data_imdc_2026/map_regional_health.csv
    map_path = "data/data_imdc_2026/map_regional_health.csv"
    if not os.path.exists(map_path):
        # Fallback search path in project subdirectories
        map_path = "../Dengue Forecasting Hackathon 2026/data/data_imdc_2026/map_regional_health.csv"
        if not os.path.exists(map_path):
            raise FileNotFoundError("Could not locate 'map_regional_health.csv' in standard paths.")

    df_map = pd.read_csv(map_path)
    state_to_code = df_map[['uf', 'uf_code']].drop_duplicates().set_index('uf')['uf_code'].to_dict()

    print(f"Starting submission for model {args.repository} (Commit: {commit_hash})")
    if args.dry_run:
        print("[DRY-RUN] Execution checks passed. Local validations succeeded. No API calls will be made.")
        return

    # Loop through states and upload
    for state_uf, state_code in state_to_code.items():
        state_pred = pred_df[pred_df['uf'] == state_uf].copy()
        if len(state_pred) == 0:
            print(f"No predictions found in input file for state {state_uf}. Skipping.")
            continue

        # Format dates as YYYY-MM-DD
        state_pred['date'] = pd.to_datetime(state_pred['date']).dt.strftime('%Y-%m-%d')

        # Sort values chronologically to ensure continuous sequence validation
        state_pred = state_pred.sort_values('date')

        # Drop the 'uf' column so it matches the api schema exactly
        state_upload_data = state_pred.drop(columns=['uf'])

        print(f"--> Uploading {state_uf} (Geocode {state_code}) containing {len(state_upload_data)} rows...")
        try:
            res = upload_prediction(
                api_key=api_key,
                repository=args.repository,
                description=f"{args.desc} - {state_uf}",
                commit=commit_hash,
                disease=args.disease,
                case_definition=args.case_definition,
                adm_level=1,
                adm_1=int(state_code),
                published=True,
                prediction=state_upload_data
            )
            print(f"    Success: {res}")
        except Exception as e:
            print(f"    [ERROR] Failed uploading predictions for state {state_uf}: {e}")

    print("\nSubmission loop completed.")

if __name__ == "__main__":
    main()
