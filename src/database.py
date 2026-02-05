"""
Database connection utilities for Neon.tech PostgreSQL
"""

import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()


def connect_to_database():
    """
    Establish connection to Neon.tech PostgreSQL database
    
    Returns:
        connection: psycopg2 connection object
    
    Raises:
        Exception: If connection fails
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT', 5432)
        )
        print("✅ Successfully connected to database")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise


def fetch_training_data(table_name='robot_current_data'):
    """
    Fetch training data from PostgreSQL database
    
    Args:
        table_name (str): Name of the table containing robot data
        
    Returns:
        pd.DataFrame: Training data with columns [Time, Axis_1, ..., Axis_8]
    """
    try:
        conn = connect_to_database()
        
        # Query to fetch all data ordered by timestamp
        query = f"SELECT * FROM {table_name} ORDER BY time_seconds"
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"✅ Fetched {len(df)} records from database")
        print(f"📊 Columns: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        raise


def create_table_if_not_exists():
    """
    Create robot_current_data table if it doesn't exist
    This is useful for initial setup
    """
    conn = connect_to_database()
    cursor = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS robot_current_data (
        id SERIAL PRIMARY KEY,
        time_seconds FLOAT NOT NULL,
        axis_1 FLOAT,
        axis_2 FLOAT,
        axis_3 FLOAT,
        axis_4 FLOAT,
        axis_5 FLOAT,
        axis_6 FLOAT,
        axis_7 FLOAT,
        axis_8 FLOAT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    conn.close()
    
    print("✅ Table created/verified successfully")


def upload_csv_to_database(csv_path, table_name='robot_current_data'):
    """
    Upload CSV data to PostgreSQL database
    
    Args:
        csv_path (str): Path to CSV file
        table_name (str): Target table name
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"📁 Loaded {len(df)} records from {csv_path}")
        
        # Connect to database
        conn = connect_to_database()
        
        # Upload data
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"✅ Successfully uploaded data to {table_name}")
        
    except Exception as e:
        print(f"❌ Error uploading CSV: {e}")
        raise


def test_connection():
    """
    Test database connection and print basic info
    """
    try:
        conn = connect_to_database()
        cursor = conn.cursor()
        
        # Get PostgreSQL version
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"📌 PostgreSQL version: {version[0]}")
        
        # List tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        print(f"📊 Available tables: {[t[0] for t in tables]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connection when run directly
    print("🔧 Testing database connection...")
    test_connection()
