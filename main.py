from pathlib import Path
from cohort_builder import CohortBuilder
import json 

# Logger Pathing Information
#############################################################
# Define log directory and log file path
log_dir = Path(__file__).resolve().parent.parent / "Output"
log_file = log_dir / "processing_info.log"
# Remove existing log file if it exists
if log_file.exists():
    log_file.unlink()
    print(f"Removed existing log file: {log_file}")

### Initialize Cohort Builder ###
#############################################################
# Note: Path can change based on where the database is located.
db_path = str(Path(__file__).resolve().parent.parent / "DE_Challenge_DB.sqlite")
cohort_builder = CohortBuilder(db_path, log_file=str(log_file))

# Investigate data
investigation_summary = cohort_builder.investigate_data()

# Generate cohort
cohort_builder.get_cohort()
