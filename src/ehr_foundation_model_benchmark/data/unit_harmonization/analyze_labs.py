import gc
import pymssql
import pandas as pd
from tqdm import tqdm


def create_measurement_unit_counts(
        server: str,
        user: str,
        password: str,
        database: str,
        output_path: str = 'measurement_unit_counts.csv'
):
    """
    Generate a CSV with measurement unit counts from OMOP database.

    Args:
        server (str): Database server name or IP
        user (str): Database username
        password (str): Database password
        database (str): Database name
        output_path (str, optional): Path to save output CSV. Defaults to 'measurement_unit_counts.csv'
    """
    # Establish database connection
    try:
        conn = pymssql.connect(
            server=server,
            user=user,
            password=password,
            database=database
        )
        print("Connection successful!")
    except Exception as e:
        print(f"Database connection error: {e}")
        return

    # SQL query to get counts per unit per measurement
    labs_query = """
    SELECT DISTINCT 
        measurement_concept_id, 
        c1.concept_name as measurement_name, 
        unit_concept_id, 
        c2.concept_name as unit_concept_name, 
        COUNT(measurement_id) as counts
    FROM dbo.measurement
    LEFT JOIN dbo.concept as c1 on c1.concept_id = measurement_concept_id
    LEFT JOIN dbo.concept as c2 on c2.concept_id = unit_concept_id
    GROUP BY 
        measurement_concept_id, 
        c1.concept_name, 
        unit_concept_id, 
        c2.concept_name
    """

    try:
        # Read SQL in chunks to handle large datasets
        list_chunks = []
        for chunk in tqdm(pd.read_sql(labs_query, conn, chunksize=10_000_000)):
            list_chunks.append(chunk)

        # Combine chunks
        df_labs = pd.concat(list_chunks)

        # Clean up memory
        del list_chunks
        gc.collect()

        # Save to CSV
        df_labs.to_csv(output_path, index=False)
        print(f"Measurement unit counts saved to {output_path}")

    except Exception as e:
        print(f"Error processing data: {e}")

    finally:
        # Close database connection
        conn.close()


def main():
    # Example usage
    create_measurement_unit_counts(
        server='your_server_name_or_ip',
        user='your_username',
        password='your_password',
        database='your_database'
    )


if __name__ == "__main__":
    main()