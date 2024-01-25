import os
import csv

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_split(pose: str):
    if pose == "19097be0e2094c4aa6b2fdc208c8231e.pose":
        return "test"

    rand = hash(pose[:-len('.pose')]) % 100
    if rand > 98:
        return "test"

    if rand > 96:
        return "dev"

    return "train"


if __name__ == "__main__":
    database = psycopg2.connect(
        dbname=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASS'],
        host=os.environ['DB_HOST']
    )

    QUERY = """
    SELECT CONCAT("videoId", '.pose') as pose, "videoLanguage", start, "end", "text" 
    FROM captions 
    WHERE language = 'Sgnw'
    """

    cursor = database.cursor()
    with open('data.csv', 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        cursor.execute(QUERY)
        writer.writerow([c.name for c in cursor.description] + ["split"])
        num_rows = cursor.rowcount
        for row in cursor:
            writer.writerow(list(row) + [get_split(row[0])])
