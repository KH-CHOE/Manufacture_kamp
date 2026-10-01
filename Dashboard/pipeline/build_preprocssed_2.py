#!/usr/bin/env python3
"""Add 2021 calendar flags, then remove date and hourly average."""
import csv
from datetime import datetime
from pathlib import Path


DATA = Path(__file__).resolve().parent.parent / "data" / "generated"
SOURCE = DATA / "okm_augumented_2021_preprocssed.csv"
OUTPUT = DATA / "okm_augumented_2021_preprocssed_2.csv"

# Korean public holidays that fell on weekdays within this dataset's date range.
# Sources: 2021 Korean calendar announcement and MPM's 2021 substitute holiday notice.
# https://www.korea.kr/news/policyNewsView.do?newsId=148873429
# https://www.mpm.go.kr/mpm/comm/newsPress/newsPressRelease/?boardId=bbs_0000000000000029&category=&cntId=3237&mode=view&pageIdx=9
WEEKDAY_HOLIDAYS = {
    "20210101",  # New Year's Day
    "20210211",  # Lunar New Year holiday
    "20210212",  # Lunar New Year
    "20210301",  # Independence Movement Day
    "20210505",  # Children's Day
    "20210519",  # Buddha's Birthday
    "20210816",  # Substitute holiday for Liberation Day (Aug 15, Sunday)
}

with SOURCE.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    original_columns = reader.fieldnames
    assert original_columns is not None
    assert "날짜" in original_columns and "평균" in original_columns
    rows = list(reader)

assert len(rows) == 24480
assert rows[0]["날짜"] == "20210101" and rows[-1]["날짜"] == "20210914"

columns = ["주중여부", "주중공휴일여부"] + [
    column for column in original_columns if column not in {"날짜", "평균"}
]
seen_holidays = set()
for row in rows:
    date = row["날짜"]
    weekday = datetime.strptime(date, "%Y%m%d").weekday() < 5
    holiday = weekday and date in WEEKDAY_HOLIDAYS
    row["주중여부"] = "1" if weekday else "0"
    row["주중공휴일여부"] = "1" if holiday else "0"
    if holiday:
        seen_holidays.add(date)
    del row["날짜"]
    del row["평균"]

assert seen_holidays == WEEKDAY_HOLIDAYS
assert all(not (row["주중공휴일여부"] == "1" and row["주중여부"] == "0") for row in rows)
assert all(set(row) == set(columns) for row in rows)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"created {OUTPUT} ({len(rows):,} rows, {len(columns)} columns)")
print("weekday public holidays:", ", ".join(sorted(seen_holidays)))
