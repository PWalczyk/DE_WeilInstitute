import sqlite3
import pandas as pd
from typing import Dict, Tuple, Any
from datetime import datetime, timedelta
from logger import Logger
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class CohortBuilder:
    """
    This is a class Primarily for Databse Investigation, Preprocessing, and Cohort building. 
    The output data from this will be used for analyses in separate python scripts.
    Handles data preprocessing and Cohort generation without modifying the original database.
    Since Databases can change, it pulls fresh data each time.
    """

    def __init__(self, db_path: str, log_file: str = "cohort_builder.log") -> None:
        """Initializes database connection and logging."""
        # Initialize Logger First
        self.logger: Logger = Logger(log_file)
        self.logger.info("\n--- Cohort Builder Logger Initialized ---")

        # Initialize Database Connection
        self.db_path: str = db_path
        self.conn: sqlite3.Connection = sqlite3.connect(db_path)
        self.cursor: sqlite3.Cursor = self.conn.cursor()
        self.logger.info("\n--- Cohort Builder Connceted to DataBase Successfully. ---")
            

    def investigate_data(self) -> Dict[str, Any]:
        """Investigates missing values, duplicates, and foreign key relationships in all tables."""
        self.logger.info("\n--- Data Investigation ---")

        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in self.cursor.fetchall()]

        data_summary: Dict[str, Any] = {}
        primary_key_summary: Dict[str, Any] = {}
        foreign_key_summary: Dict[str, Any] = {}

        for table in tables:
            self.logger.info(f"\nAnalyzing Table: {table}")

            # Fetch table schema to get column names
            schema_query = f"PRAGMA table_info({table});"
            schema_df = pd.read_sql_query(schema_query, self.conn)

            # Identify primary keys
            primary_keys = schema_df[schema_df["pk"] > 0][["name"]].to_dict(orient="records")
            primary_key_summary[table] = [pk["name"] for pk in primary_keys] if primary_keys else "None"
            self.logger.info(f"Primary Keys in {table}: {primary_key_summary[table]}")

            if schema_df.empty:
                self.logger.info(f"Skipping empty table: {table}")
                continue

            column_names = schema_df["name"].tolist()

            # Fetch entire table into Pandas DataFrame
            df = pd.read_sql_query(f"SELECT * FROM {table}", self.conn)

            # Count missing values per column
            missing_values = df.isnull().sum()
            missing_summary = missing_values[missing_values > 0]  # Filter only columns with missing values

            # Count duplicate rows
            duplicate_count = df.duplicated().sum()

            # Fetch foreign key constraints
            foreign_keys = pd.read_sql_query(f"PRAGMA foreign_key_list({table});", self.conn)
            if not foreign_keys.empty:
                fk_mappings = foreign_keys[["from", "table", "to"]].to_dict(orient="records")
                foreign_key_summary[table] = fk_mappings
                self.logger.info(f"Foreign Key Relationships in {table}: {fk_mappings}")

            # Log findings
            self.logger.info("\nMissing Values:\n" + (missing_summary.to_string() if not missing_summary.empty else "None"))
            self.logger.info(f"\nDuplicate Rows: {duplicate_count}")

            # Store results
            data_summary[table] = {
                "primary_keys": primary_key_summary[table],
                "missing_values": missing_summary.to_dict(),
                "duplicates": duplicate_count,
                "foreign_keys": fk_mappings if not foreign_keys.empty else "None"
            }
        self.logger.info("\n Data_summary: " + str(data_summary))
        # Log overall foreign key summary
        self.logger.info("\n--- Foreign Key Summary Across Tables ---")
        self.logger.info(str(foreign_key_summary) if foreign_key_summary else "No foreign keys found.")

        return data_summary


    def get_hf_patients(self) -> pd.DataFrame:
        """Retrieve HF & Cardiac Dysfunction patients admitted in June 2000."""
        query = """
        WITH Discharges AS (
            -- Precompute the latest discharge date per patient to avoid redundant subqueries
            SELECT t.SUBJECT_ID, t.HADM_ID, MAX(t.INTIME) AS DISCHARGE_DATE
            FROM TRANSFERS t
            WHERE t.EVENTTYPE = 'discharge'
            GROUP BY t.SUBJECT_ID, t.HADM_ID
        ),
        TroponinPatients AS (
            -- Get all patients who had at least one Troponin test during their admission
            SELECT DISTINCT l.SUBJECT_ID, l.HADM_ID
            FROM LABS l
            JOIN ICD_LABS il ON l.ITEMID = il.ITEMID
            WHERE il.LABEL LIKE '%Troponin%'
        )
        SELECT DISTINCT d.SUBJECT_ID, d.HADM_ID, d.ICD9_CODE, i.LONG_TITLE AS ICD_DESCRIPTION,
            t.CURR_CAREUNIT, DATETIME(t.INTIME) AS ADMISSION_DATE, 
            DATETIME(ds.DISCHARGE_DATE) AS DISCHARGE_DATE
        FROM DIAGNOSES d
        LEFT JOIN ICD_DIAGNOSES i ON d.ICD9_CODE = i.ICD9_CODE
        LEFT JOIN TRANSFERS t ON d.SUBJECT_ID = t.SUBJECT_ID AND d.HADM_ID = t.HADM_ID
        LEFT JOIN Discharges ds ON d.SUBJECT_ID = ds.SUBJECT_ID AND d.HADM_ID = ds.HADM_ID
        INNER JOIN TroponinPatients tp ON d.SUBJECT_ID = tp.SUBJECT_ID AND d.HADM_ID = tp.HADM_ID
        WHERE (d.ICD9_CODE LIKE '428%' OR d.ICD9_CODE LIKE '425%' OR d.ICD9_CODE = '4299' OR d.ICD9_CODE = '78551')
        AND t.EVENTTYPE = 'admit'
        AND strftime('%Y-%m', t.INTIME) = '2000-06'
        ORDER BY d.SUBJECT_ID, ADMISSION_DATE;
        """
        self.logger.info("\n Querying Database for HF/CD Patients Admitted in June 2000")
        df = pd.read_sql_query(query, self.conn)
        self.logger.info(f"\n Retrieved {len(df['SUBJECT_ID'].unique())} HF/CD patients admitted in June 2000")
        return df


    def get_nurses(self) -> pd.DataFrame:
        """Retrieve all nurse assignments per unit."""
        query = """
        SELECT DISTINCT t.CURR_CAREUNIT, t.HADM_ID, n.CGID, c.LABEL, 
               DATETIME(n.STARTTIME) AS STARTTIME, DATETIME(n.ENDTIME) AS ENDTIME
        FROM TRANSFERS t
        LEFT JOIN TREATMENT_TEAM n ON t.HADM_ID = n.HADM_ID
        LEFT JOIN CARE_GIVERS c ON n.CGID = c.CGID
        WHERE c.LABEL LIKE '%RN%';
        """
        self.logger.info("\n Querying Database for Nurse Assignments")
        df = pd.read_sql_query(query, self.conn)
        self.logger.info(f"\n Retrieved {len(df)} nurse assignments across units")
        return df


    def generate_time_blocks(self, patients_df: pd.DataFrame, nurses_df: pd.DataFrame) -> tuple:
        """
        Expands each patient’s admission period and each nurse’s shift into 12-hour time blocks, 
        with nested 4-hour sub-blocks. Retains admission (`INTIME`) and discharge (`DISCHARGE_DATE`) times.

        Parameters:
        - patients_df (pd.DataFrame): DataFrame containing patient admissions, with columns:
            - SUBJECT_ID (int)
            - HADM_ID (int)
            - CURR_CAREUNIT (str)
            - ADMISSION_DATE (datetime) [Previously `INTIME`]
            - DISCHARGE_DATE (datetime)
        - nurses_df (pd.DataFrame): DataFrame containing nurse shift data, with columns:
            - CGID (int) - Nurse ID
            - CURR_CAREUNIT (str) - Unit the nurse worked in
            - STARTTIME (datetime) - Shift start time
            - ENDTIME (datetime) - Shift end time

        Returns:
        - tuple(pd.DataFrame, pd.DataFrame): 
            - Expanded patient DataFrame with **12-hour and 4-hour time blocks, keeping `ADMISSION_DATE` and `DISCHARGE_DATE`**.
            - Expanded nurse DataFrame with **12-hour and 4-hour time blocks**.
        """
        self.logger.info("\n Generating 12-hour Time Blocks for Patients and Nurses")
        # Ensure proper datetime format
        patients_df["ADMISSION_DATE"] = pd.to_datetime(patients_df["ADMISSION_DATE"])
        patients_df["DISCHARGE_DATE"] = pd.to_datetime(patients_df["DISCHARGE_DATE"])
        nurses_df["STARTTIME"] = pd.to_datetime(nurses_df["STARTTIME"])
        nurses_df["ENDTIME"] = pd.to_datetime(nurses_df["ENDTIME"])

        # Generate all 12-hour blocks in June 2000
        start_time = datetime(2000, 6, 1, 0, 0)
        end_time = datetime(2000, 6, 30, 23, 59)
        twelve_hour_blocks = [start_time + timedelta(hours=12 * i) for i in range((end_time - start_time).days * 2 + 1)]

        # Define the 4-hour sub-block offsets relative to each 12-hour block
        sub_block_offsets = [timedelta(hours=0), timedelta(hours=4), timedelta(hours=8)]

        expanded_patient_rows = []
        expanded_nurse_rows = []

        # Process patient time blocks (keeping ADMISSION_DATE and DISCHARGE_DATE)
        for _, row in patients_df.iterrows():
            subject_id = row["SUBJECT_ID"]
            hadm_id = row["HADM_ID"]
            curr_careunit = row["CURR_CAREUNIT"]
            admit_time = row["ADMISSION_DATE"]
            discharge_time = row["DISCHARGE_DATE"] if pd.notna(row["DISCHARGE_DATE"]) else end_time

            # Assign patient to all 12-hour blocks within admission-discharge window
            for twelve_hour_start in twelve_hour_blocks:
                if admit_time <= twelve_hour_start < discharge_time:
                    # Generate corresponding 4-hour sub-blocks
                    for offset in sub_block_offsets:
                        sub_block_start = twelve_hour_start + offset
                        expanded_patient_rows.append([
                            subject_id, hadm_id, curr_careunit, 
                            twelve_hour_start, sub_block_start, admit_time, discharge_time
                        ])

        # Process nurse time blocks
        for _, row in nurses_df.iterrows():
            cgid = row["CGID"]
            curr_careunit = row["CURR_CAREUNIT"]
            start_time_nurse = row["STARTTIME"]
            end_time_nurse = row["ENDTIME"] if pd.notna(row["ENDTIME"]) else end_time

            # Assign nurse to all 12-hour blocks within shift window
            for twelve_hour_start in twelve_hour_blocks:
                if start_time_nurse <= twelve_hour_start < end_time_nurse:
                    # Generate corresponding 4-hour sub-blocks
                    for offset in sub_block_offsets:
                        sub_block_start = twelve_hour_start + offset
                        expanded_nurse_rows.append([cgid, curr_careunit, twelve_hour_start, sub_block_start])

        # Create DataFrames
        expanded_patients_df = pd.DataFrame(expanded_patient_rows, columns=[
            "SUBJECT_ID", "HADM_ID", "CURR_CAREUNIT", "TIME_BLOCK_12HR", "TIME_BLOCK_4HR", "ADMISSION_DATE", "DISCHARGE_DATE"
        ])
        
        expanded_nurses_df = pd.DataFrame(expanded_nurse_rows, columns=[
            "CGID", "CURR_CAREUNIT", "TIME_BLOCK_12HR", "TIME_BLOCK_4HR"
        ])

        self.logger.info("\n Successfully Generated 12-hour Time Blocks for Patients and Nurses")
        self.logger.info(f"\n Generated {len(expanded_patients_df)} time-block entries for {patients_df.shape[0]} patients.")
        self.logger.info(f"\n Generated {len(expanded_nurses_df)} time-block entries for {nurses_df.shape[0]} nurses.")


        return expanded_patients_df, expanded_nurses_df


    def compute_discharge_to_nurse_ratio(self, patients_df: pd.DataFrame, nurses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the discharge-to-nurse ratio for each 12-hour block using pre-generated 4-hour sub-blocks.

        Parameters:
        - patients_df (pd.DataFrame): Expanded patient DataFrame with `TIME_BLOCK_12HR` and `TIME_BLOCK_4HR`, including:
            - SUBJECT_ID, HADM_ID, CURR_CAREUNIT, TIME_BLOCK_12HR, TIME_BLOCK_4HR, ADMISSION_DATE, DISCHARGE_DATE
        - nurses_df (pd.DataFrame): Expanded nurse DataFrame with `TIME_BLOCK_12HR` and `TIME_BLOCK_4HR`.

        Returns:
        - pd.DataFrame: DataFrame with discharge-to-nurse ratio metrics for each 12-hour block.
        """
        self.logger.info("\n Computing Discharge-to-Nurse Ratio for 12-hour Blocks using 4-hour Sub-blocks")
        # Ensure all date columns are in datetime format
        patients_df["TIME_BLOCK_12HR"] = pd.to_datetime(patients_df["TIME_BLOCK_12HR"])
        patients_df["TIME_BLOCK_4HR"] = pd.to_datetime(patients_df["TIME_BLOCK_4HR"])
        patients_df["DISCHARGE_DATE"] = pd.to_datetime(patients_df["DISCHARGE_DATE"])
        nurses_df["TIME_BLOCK_12HR"] = pd.to_datetime(nurses_df["TIME_BLOCK_12HR"])
        nurses_df["TIME_BLOCK_4HR"] = pd.to_datetime(nurses_df["TIME_BLOCK_4HR"])

        # Identify the first 4-hour block in which each patient was discharged (within the correct 12-hour period)
        patients_df["DISCHARGE_TIME_BLOCK"] = patients_df.groupby(["SUBJECT_ID", "HADM_ID", "TIME_BLOCK_12HR"])["TIME_BLOCK_4HR"].transform(
            lambda x: x[x >= patients_df.loc[x.index, "DISCHARGE_DATE"]].min()
        )

        # Count discharges per unit per 4-hour sub-block (only counting each patient once within the 12-hour period)
        discharges = (
            patients_df[patients_df["TIME_BLOCK_4HR"] == patients_df["DISCHARGE_TIME_BLOCK"]]
            .groupby(["CURR_CAREUNIT", "TIME_BLOCK_12HR", "TIME_BLOCK_4HR"])
            .size()
            .reset_index(name="num_discharged")
        )

        # Count nurses per unit per 4-hour sub-block
        nurses = (
            nurses_df
            .groupby(["CURR_CAREUNIT", "TIME_BLOCK_12HR", "TIME_BLOCK_4HR"])
            .size()
            .reset_index(name="num_nurses")
        )

        # Merge discharges & nurses
        merged = pd.merge(discharges, nurses, on=["CURR_CAREUNIT", "TIME_BLOCK_12HR", "TIME_BLOCK_4HR"], how="outer").fillna(0)

        # Compute discharge-to-nurse ratio (handling divide by zero)
        merged["DISCHARGE_NURSE_RATIO"] = merged["num_discharged"] / merged["num_nurses"]
        merged["DISCHARGE_NURSE_RATIO"].replace([float("inf"), -float("inf")], 0, inplace=True)  # Handle divide by zero

        # Normalize the Data since we get varying number of discharges and nurses
        merged["UNIT_NORM_RATIO"] = merged.groupby("CURR_CAREUNIT")["DISCHARGE_NURSE_RATIO"].transform(lambda x: x / x.mean())

        # Aggregate to 12-hour summary
        summary_df = (
            merged
            .groupby(["CURR_CAREUNIT", "TIME_BLOCK_12HR"])
            .agg(
                DISCHARGE_4HR_1=("num_discharged", lambda x: x.iloc[0] if len(x) > 0 else 0),
                DISCHARGE_4HR_2=("num_discharged", lambda x: x.iloc[1] if len(x) > 1 else 0),
                DISCHARGE_4HR_3=("num_discharged", lambda x: x.iloc[2] if len(x) > 2 else 0),
                NURSES_4HR_1=("num_nurses", lambda x: x.iloc[0] if len(x) > 0 else 0),
                NURSES_4HR_2=("num_nurses", lambda x: x.iloc[1] if len(x) > 1 else 0),
                NURSES_4HR_3=("num_nurses", lambda x: x.iloc[2] if len(x) > 2 else 0),
                DISCHARGE_NURSE_RATIO_4HR_1=("DISCHARGE_NURSE_RATIO", lambda x: x.iloc[0] if len(x) > 0 else 0),
                DISCHARGE_NURSE_RATIO_4HR_2=("DISCHARGE_NURSE_RATIO", lambda x: x.iloc[1] if len(x) > 1 else 0),
                DISCHARGE_NURSE_RATIO_4HR_3=("DISCHARGE_NURSE_RATIO", lambda x: x.iloc[2] if len(x) > 2 else 0),
                UNIT_NORM_RATIO_4HR_1=("UNIT_NORM_RATIO", lambda x: x.iloc[0] if len(x) > 0 else 0),
                UNIT_NORM_RATIO_4HR_2=("UNIT_NORM_RATIO", lambda x: x.iloc[1] if len(x) > 1 else 0),
                UNIT_NORM_RATIO_4HR_3=("UNIT_NORM_RATIO", lambda x: x.iloc[2] if len(x) > 2 else 0),
            )
            .reset_index()
        )

        # Compute the final averaged ratio across all 3 sub-blocks
        summary_df["AVG_DISCHARGE_TO_NURSE_RATIO"] = summary_df[
            ["DISCHARGE_NURSE_RATIO_4HR_1", "DISCHARGE_NURSE_RATIO_4HR_2", "DISCHARGE_NURSE_RATIO_4HR_3"]
        ].mean(axis=1)

        summary_df["AVG_UNIT_NORM_RATIO"] = summary_df[
            ["UNIT_NORM_RATIO_4HR_1", "UNIT_NORM_RATIO_4HR_2", "UNIT_NORM_RATIO_4HR_3"]
        ].mean(axis=1)

        self.logger.info("\n Successfully Computed Discharge-to-Nurse Ratio for 12-hour Blocks using 4-hour Sub-blocks")
        self.logger.info(f"\n Computed discharge-to-nurse ratio for {summary_df.shape[0]} 12-hour blocks.")
        return summary_df


    def compute_nurse_to_patient_ratio(self, patients_df: pd.DataFrame, nurses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the nurse-to-patient ratio for each 12-hour block using pre-generated 4-hour sub-blocks.

        Parameters:
        - patients_df (pd.DataFrame): Expanded patient DataFrame with `TIME_BLOCK_12HR` and `TIME_BLOCK_4HR`, including:
            - SUBJECT_ID, HADM_ID, CURR_CAREUNIT, TIME_BLOCK_12HR, TIME_BLOCK_4HR, ADMISSION_DATE, DISCHARGE_DATE
        - nurses_df (pd.DataFrame): Expanded nurse DataFrame with `TIME_BLOCK_12HR` and `TIME_BLOCK_4HR`.

        Returns:
        - pd.DataFrame: DataFrame with nurse-to-patient ratio metrics for each 12-hour block.
        """
        self.logger.info("\n Computing Nurse-to-Patient Ratio for 12-hour Blocks using 4-hour Sub-blocks")
        # Ensure all date columns are in datetime format
        patients_df["TIME_BLOCK_12HR"] = pd.to_datetime(patients_df["TIME_BLOCK_12HR"])
        patients_df["TIME_BLOCK_4HR"] = pd.to_datetime(patients_df["TIME_BLOCK_4HR"])
        patients_df["DISCHARGE_DATE"] = pd.to_datetime(patients_df["DISCHARGE_DATE"])
        nurses_df["TIME_BLOCK_12HR"] = pd.to_datetime(nurses_df["TIME_BLOCK_12HR"])
        nurses_df["TIME_BLOCK_4HR"] = pd.to_datetime(nurses_df["TIME_BLOCK_4HR"])

        # Count the total number of patients per unit per 4-hour sub-block
        patients_in_unit = (
            patients_df[
                (patients_df["TIME_BLOCK_4HR"] >= patients_df["ADMISSION_DATE"]) & 
                (patients_df["TIME_BLOCK_4HR"] < patients_df["DISCHARGE_DATE"])
            ]
            .groupby(["CURR_CAREUNIT", "TIME_BLOCK_12HR", "TIME_BLOCK_4HR"])
            .size()
            .reset_index(name="num_patients")
        )

        # Count the number of nurses per unit per 4-hour sub-block
        nurses_in_unit = (
            nurses_df
            .groupby(["CURR_CAREUNIT", "TIME_BLOCK_12HR", "TIME_BLOCK_4HR"])
            .size()
            .reset_index(name="num_nurses")
        )

        # Merge patient and nurse counts
        merged = pd.merge(patients_in_unit, nurses_in_unit, on=["CURR_CAREUNIT", "TIME_BLOCK_12HR", "TIME_BLOCK_4HR"], how="outer").fillna(0)

        # Compute nurse-to-patient ratio (set to 0 if no patients)
        # Replace any merged["num_patients"] values with 1 to avoid divide by zero
        merged["num_patients"].replace(0, 1, inplace=True)
        merged["nurse_to_patient_ratio"] = merged["num_nurses"] / merged["num_patients"]
        merged["nurse_to_patient_ratio"].replace([float("inf"), -float("inf")], 0, inplace=True)  # Handle divide by zero

        # Normalize the Data since we get varying number of patients and nurses
        merged["unit_norm_ratio"] = merged.groupby("CURR_CAREUNIT")["nurse_to_patient_ratio"].transform(lambda x: x / x.mean())

        # Aggregate into a 12-hour summary
        summary_df = (
            merged
            .groupby(["CURR_CAREUNIT", "TIME_BLOCK_12HR"])
            .agg(
                PATIENTS_4HR_1=("num_patients", lambda x: x.iloc[0] if len(x) > 0 else 0),
                PATIENTS_4HR_2=("num_patients", lambda x: x.iloc[1] if len(x) > 1 else 0),
                PATIENTS_4HR_3=("num_patients", lambda x: x.iloc[2] if len(x) > 2 else 0),
                NURSES_4HR_1=("num_nurses", lambda x: x.iloc[0] if len(x) > 0 else 0),
                NURSES_4HR_2=("num_nurses", lambda x: x.iloc[1] if len(x) > 1 else 0),
                NURSES_4HR_3=("num_nurses", lambda x: x.iloc[2] if len(x) > 2 else 0),
                NURSE_TO_PATIENT_RATIO_4HR_1=("nurse_to_patient_ratio", lambda x: x.iloc[0] if len(x) > 0 else 0),
                NURSE_TO_PATIENT_RATIO_4HR_2=("nurse_to_patient_ratio", lambda x: x.iloc[1] if len(x) > 1 else 0),
                NURSE_TO_PATIENT_RATIO_4HR_3=("nurse_to_patient_ratio", lambda x: x.iloc[2] if len(x) > 2 else 0),
                UNIT_NORM_RATIO_4HR_1=("unit_norm_ratio", lambda x: x.iloc[0] if len(x) > 0 else 0),
                UNIT_NORM_RATIO_4HR_2=("unit_norm_ratio", lambda x: x.iloc[1] if len(x) > 1 else 0),
                UNIT_NORM_RATIO_4HR_3=("unit_norm_ratio", lambda x: x.iloc[2] if len(x) > 2 else 0),
            )
            .reset_index()
        )

        # Compute the final averaged ratio across all 3 sub-blocks
        summary_df["AVG_NURSE_TO_PATIENT_RATIO"] = summary_df[
            ["NURSE_TO_PATIENT_RATIO_4HR_1", "NURSE_TO_PATIENT_RATIO_4HR_2", "NURSE_TO_PATIENT_RATIO_4HR_3"]
        ].mean(axis=1)

        summary_df["AVG_UNIT_NORM_RATIO"] = summary_df[
            ["UNIT_NORM_RATIO_4HR_1", "UNIT_NORM_RATIO_4HR_2", "UNIT_NORM_RATIO_4HR_3"]
        ].mean(axis=1)

        self.logger.info("\n Successfully Computed Nurse-to-Patient Ratio for 12-hour Blocks using 4-hour Sub-blocks")
        self.logger.info(f"\n Computed nurse-to-patient ratio for {summary_df.shape[0]} 12-hour blocks.")
        return summary_df


    def subset_labs_for_cohort(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Subsets the `LABS` table to only include Troponin & BNP tests for patients in the cohort,
        ensuring that test results are assigned to the correct admission.

        Parameters:
        - db_conn (sqlite3.Connection): Database connection to execute SQL queries.
        - patients_df (pd.DataFrame): Cohort patient DataFrame, including:
            - SUBJECT_ID, HADM_ID, ADMISSION_DATE, DISCHARGE_DATE

        Returns:
        - pd.DataFrame: Filtered lab results for cohort patients, including:
            - SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOFM, LABEL (from ICD_LABS)
        """
        self.logger.info("\n Subsetting Troponin & BNP Tests for Cohort Patients")

        # Convert HADM_IDs to a plain Python list of integers
        hadm_ids = list(map(int, patients_df["HADM_ID"].unique()))  # Avoid np.int64 issues

        # If too many HADM_IDs, split into chunks (SQLite IN limit is around 999)
        chunk_size = 500  # Safe limit
        hadm_id_chunks = [hadm_ids[i:i + chunk_size] for i in range(0, len(hadm_ids), chunk_size)]

        # Initialize empty DataFrame to store results
        labs_subset_df = pd.DataFrame()

        # Loop through chunks and execute separate queries
        for chunk in hadm_id_chunks:
            query = f"""
            SELECT l.SUBJECT_ID, l.HADM_ID, l.ITEMID, l.CHARTTIME, l.VALUENUM, l.VALUEUOM, il.LABEL
            FROM LABS l
            JOIN ICD_LABS il ON l.ITEMID = il.ITEMID
            WHERE (il.LABEL LIKE '%Troponin%' OR il.LABEL LIKE '%BNP%')
            AND l.HADM_ID IN ({','.join(['?']*len(chunk))})
            """
            
            # Fetch data from SQL using parameterized query
            chunk_df = pd.read_sql_query(query, self.conn, params=chunk)

            # Append to the main DataFrame
            labs_subset_df = pd.concat([labs_subset_df, chunk_df], ignore_index=True)

        # Ensure CHARTTIME is in datetime format
        labs_subset_df["CHARTTIME"] = pd.to_datetime(labs_subset_df["CHARTTIME"])

        # Merge with patients_df to get admission and discharge dates
        labs_subset_df = labs_subset_df.merge(
            patients_df[["SUBJECT_ID", "HADM_ID", "ADMISSION_DATE", "DISCHARGE_DATE"]],
            on=["SUBJECT_ID", "HADM_ID"],
            how="inner"
        )

        # Keep only lab results within the correct admission period
        labs_subset_df = labs_subset_df[
            (labs_subset_df["CHARTTIME"] >= labs_subset_df["ADMISSION_DATE"]) &
            (labs_subset_df["CHARTTIME"] < labs_subset_df["DISCHARGE_DATE"])
        ]


        self.logger.info("\n Successfully Subsetting Troponin & BNP Tests for Cohort Patients")
        self.logger.info(f"\n Loaded {labs_subset_df.shape[0]} Troponin & BNP test results for cohort patients.")
        return labs_subset_df


    def compute_latest_lab_test(self, patients_df: pd.DataFrame, labs_subset_df: pd.DataFrame, test_name: str, value_col_prefix: str) -> pd.DataFrame:
        """
        Retrieves the most recent lab test values per patient per 4-hour block,
        aggregates to find the latest per 12-hour block, and fills forward if no new test exists.

        Parameters:
        - patients_df (pd.DataFrame): DataFrame containing patient time blocks.
        - labs_subset_df (pd.DataFrame): DataFrame containing lab test results.
        - test_name (str): The name (or substring) of the test to filter (e.g., "Troponin" or "BNP").
        - value_col_prefix (str): The prefix for renaming value columns (e.g., "TROPONIN" or "BNP").

        Returns:
        - pd.DataFrame: DataFrame with latest test values per 12-hour block.
        """
        self.logger.info(f"\n Computing Latest {test_name} Values for 12-hour Blocks for Cohort Patients")

        # Ensure datetime format
        patients_df["TIME_BLOCK_12HR"] = pd.to_datetime(patients_df["TIME_BLOCK_12HR"])
        patients_df["TIME_BLOCK_4HR"] = pd.to_datetime(patients_df["TIME_BLOCK_4HR"])
        labs_subset_df["CHARTTIME"] = pd.to_datetime(labs_subset_df["CHARTTIME"])

        # Filter only relevant tests
        test_df = labs_subset_df[labs_subset_df["LABEL"].str.contains(test_name, case=False, na=False)]

        # Merge with patients to get all time blocks
        test_df = patients_df.merge(
            test_df,
            on=["SUBJECT_ID", "HADM_ID"],
            how="left"
        )

        # Keep only tests within their corresponding 4-hour block
        test_df = test_df[
            (test_df["CHARTTIME"] >= test_df["TIME_BLOCK_4HR"]) & 
            (test_df["CHARTTIME"] < test_df["TIME_BLOCK_4HR"] + pd.Timedelta(hours=4))
        ]

        # Get the latest test per 4-hour block
        latest_4hr_test = (
            test_df.sort_values(["SUBJECT_ID", "HADM_ID", "TIME_BLOCK_4HR", "CHARTTIME"])
            .groupby(["SUBJECT_ID", "HADM_ID", "TIME_BLOCK_4HR"])
            .last()[["VALUENUM", "VALUEUOM"]]
            .reset_index()
        )

        # Rename columns
        latest_4hr_test.rename(
            columns={"VALUENUM": f"{value_col_prefix}_VAL", "VALUEUOM": f"{value_col_prefix}_UNIT"}, 
            inplace=True
        )

        # Merge with patients to ensure all time blocks exist before filling
        patients_with_test = patients_df.merge(latest_4hr_test, on=["SUBJECT_ID", "HADM_ID", "TIME_BLOCK_4HR"], how="left")

        # Forward fill values within the same admission
        patients_with_test = patients_with_test.sort_values(["SUBJECT_ID", "HADM_ID", "TIME_BLOCK_4HR"])
        patients_with_test[[f"{value_col_prefix}_VAL", f"{value_col_prefix}_UNIT"]] = (
            patients_with_test.groupby(["SUBJECT_ID", "HADM_ID"])[[f"{value_col_prefix}_VAL", f"{value_col_prefix}_UNIT"]]
            .ffill()
        )

        # Aggregate to 12-hour summary
        summary_df = (
            patients_with_test
            .sort_values(["SUBJECT_ID", "HADM_ID", "TIME_BLOCK_12HR", "TIME_BLOCK_4HR"])
            .groupby(["SUBJECT_ID", "HADM_ID", "TIME_BLOCK_12HR"])
            .last()[[f"{value_col_prefix}_VAL", f"{value_col_prefix}_UNIT"]]
            .reset_index()
        )

        self.logger.info(f"\n Computed latest {test_name} values for {summary_df.shape[0]} 12-hour blocks.")

        return summary_df


    def get_latest_test_per_patient(self, patients_df: pd.DataFrame, test_df: pd.DataFrame, test_column: str, unit_column: str) -> pd.DataFrame:
        """
        Retrieves the most recent test value (Troponin or BNP) for each patient,
        considering only their latest hospital admission.

        Parameters:
        - patients_df (pd.DataFrame): The full cohort DataFrame, including all admissions.
        - test_df (pd.DataFrame): The test-specific DataFrame (either Troponin or BNP summary).
        - test_column (str): Column name for the test value (e.g., "TROPONIN_VAL" or "BNP_VAL").
        - unit_column (str): Column name for the test unit (e.g., "TROPONIN_UNIT" or "BNP_UNIT").

        Returns:
        - pd.DataFrame: DataFrame containing the latest test value per patient.
        """
        self.logger.info(f"\n Retrieving Latest {test_column} Values for Patients")

        # Ensure date format
        patients_df["ADMISSION_DATE"] = pd.to_datetime(patients_df["ADMISSION_DATE"])
        test_df["TIME_BLOCK_12HR"] = pd.to_datetime(test_df["TIME_BLOCK_12HR"])

        # Get the most recent admission per patient
        latest_admissions = (
            patients_df.sort_values(["SUBJECT_ID", "ADMISSION_DATE"], ascending=[True, False])
            .drop_duplicates(subset=["SUBJECT_ID"], keep="first")  # Keep latest admission
            .loc[:, ["SUBJECT_ID", "HADM_ID", "ADMISSION_DATE"]]
        )

        # Merge latest admissions with the test dataset
        latest_tests = latest_admissions.merge(
            test_df,
            on=["SUBJECT_ID", "HADM_ID"],
            how="left"
        )

        # Get the latest test result per patient (most recent time block)
        latest_tests = (
            latest_tests.sort_values(["SUBJECT_ID", "TIME_BLOCK_12HR"], ascending=[True, False])
            .drop_duplicates(subset=["SUBJECT_ID"], keep="first")  # Keep latest test result
            .loc[:, ["SUBJECT_ID", test_column, unit_column]]
        )

        self.logger.info(f"\n [INFO] Retrieved latest {test_column} values for {latest_tests.shape[0]} patients.")

        return latest_tests


    def close_connection(self) -> None:
        """Closes the SQLite database connection."""
        self.conn.close()
        self.logger.info("\nDatabase connection closed.")


    def get_cohort(self) -> pd.DataFrame:
        """Main function to merge all components into final cohort data."""
        cohort_patients = self.get_hf_patients()
        nurse_assignments = self.get_nurses()

        expanded_patients_df, expanded_nurses_df = self.generate_time_blocks(cohort_patients, nurse_assignments)
        expanded_patients_df.to_csv("Output/expanded_patients.csv", index=False)
        expanded_nurses_df.to_csv("Output/expanded_nurses.csv", index=False)

        discharge_ratio = self.compute_discharge_to_nurse_ratio(expanded_patients_df, expanded_nurses_df)
        discharge_ratio.to_csv("Output/discharge_ratio.csv", index=False)

        patient_ratio = self.compute_nurse_to_patient_ratio(expanded_patients_df, expanded_nurses_df)
        patient_ratio.to_csv("Output/patient_ratio.csv", index=False)

        labs_df = self.subset_labs_for_cohort(cohort_patients)
        labs_df.to_csv("Output/labs_subset.csv", index=False)

        # lab_results_troponin = self.compute_latest_troponin(expanded_patients_df, labs_df)
        lab_results_troponin = self.compute_latest_lab_test(expanded_patients_df, labs_df, test_name="Troponin", value_col_prefix="TROPONIN")
        lab_results_troponin.to_csv("Output/lab_results_troponin.csv", index=False)

        # lab_results_bnp = self.compute_latest_bnp(expanded_patients_df, labs_df)
        lab_results_bnp = self.compute_latest_lab_test(expanded_patients_df, labs_df, test_name="BNP", value_col_prefix="BNP")
        lab_results_bnp.to_csv("Output/lab_results_bnp.csv", index=False)

        latest_troponin = self.get_latest_test_per_patient(expanded_patients_df, lab_results_troponin, "TROPONIN_VAL", "TROPONIN_UNIT")
        latest_troponin.to_csv("Output/latest_troponin.csv", index=False)

        latest_bnp = self.get_latest_test_per_patient(expanded_patients_df, lab_results_bnp, "BNP_VAL", "BNP_UNIT")
        latest_bnp.to_csv("Output/latest_bnp.csv", index=False)

        self.logger.info("\nCohort data successfully generated.")
        self.logger.info("\nClosing database connection.")

        self.close_connection()

