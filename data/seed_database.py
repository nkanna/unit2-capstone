"""
Reproducible seed script for data/database.sqlite.

Run from the project root:  python data/seed_database.py

All data is entirely fictional. The random seed is fixed so every run
produces identical rows. Drop-and-recreate makes the script idempotent.

Tables
------
customers  – 50 rows: name, region, tier, status, signup_date, mrr, industry
sales      – ~230 rows: one order per customer per active month (2024)
employees  – 40 rows: name, department, region, hire_date, annual_salary,
             satisfaction_score, status
"""

import argparse
import os
import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "database.sqlite"

# ── Reproducible randomness ────────────────────────────────────────────────
RNG = random.Random(42)

# ── Reference data ────────────────────────────────────────────────────────
REGIONS = ["North", "South", "East", "West", "EMEA"]
TIERS = ["Enterprise", "Mid-Market", "SMB"]
INDUSTRIES = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
PRODUCTS = ["Platform", "Add-on Analytics", "Professional Services"]
DEPARTMENTS = ["Engineering", "Sales", "Customer Success", "HR", "Finance"]

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe", "Aaron", "Beth", "Carlos", "Diana",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson",
    "White", "Harris", "Martin", "Thompson", "Lee", "Walker",
]

COMPANY_SUFFIXES = ["Inc", "LLC", "Corp", "Solutions", "Group", "Technologies"]

MONTHS_2024 = [f"2024-{m:02d}" for m in range(1, 13)]


def _rand_name():
    return f"{RNG.choice(FIRST_NAMES)} {RNG.choice(LAST_NAMES)}"


def _rand_company():
    word = RNG.choice(LAST_NAMES)
    suffix = RNG.choice(COMPANY_SUFFIXES)
    return f"{word} {suffix}"


def _rand_date_in_month(ym: str) -> str:
    day = RNG.randint(1, 28)
    return f"{ym}-{day:02d}"


def build_customers() -> list[tuple]:
    """50 customers: 10 per region, mixed tiers, ~20% churned."""
    rows = []
    for region in REGIONS:
        for i in range(10):
            tier = RNG.choices(TIERS, weights=[2, 4, 4])[0]
            status = "churned" if RNG.random() < 0.20 else "active"
            # MRR varies by tier
            if tier == "Enterprise":
                mrr = round(RNG.uniform(8_500, 18_000), 2)
            elif tier == "Mid-Market":
                mrr = round(RNG.uniform(1_700, 8_499), 2)
            else:
                mrr = round(RNG.uniform(100, 1_699), 2)
            signup_month = RNG.choice(MONTHS_2024[:9])  # signed up in first 9 months
            signup_date = _rand_date_in_month(signup_month)
            rows.append((
                _rand_company(),
                region,
                tier,
                status,
                signup_date,
                mrr,
                RNG.choice(INDUSTRIES),
            ))
    return rows


def build_sales(customer_rows: list[tuple]) -> list[tuple]:
    """
    Generate monthly sales rows for each customer.
    Active customers generate revenue for every month after signup.
    Churned customers stop in a randomly chosen month.
    """
    rows = []
    reps = [_rand_name() for _ in range(10)]

    for cust_id, cust in enumerate(customer_rows, start=1):
        _name, region, tier, status, signup_date, mrr, _industry = cust
        signup_month = signup_date[:7]  # "YYYY-MM"

        # Determine which months this customer paid
        active_months = [m for m in MONTHS_2024 if m >= signup_month]
        if status == "churned" and len(active_months) > 1:
            churn_idx = RNG.randint(1, len(active_months) - 1)
            active_months = active_months[:churn_idx]

        for month in active_months:
            # Add slight MRR variation (+/- 5%)
            amount = round(mrr * RNG.uniform(0.95, 1.05), 2)
            sale_date = _rand_date_in_month(month)
            product = RNG.choices(
                PRODUCTS,
                weights=[6, 2, 2] if tier == "Enterprise" else [8, 1, 1],
            )[0]
            rows.append((
                cust_id,
                amount,
                sale_date,
                region,
                RNG.choice(reps),
                product,
            ))
    return rows


def build_employees() -> list[tuple]:
    """40 employees across 5 departments, mixed regions."""
    rows = []
    for i in range(40):
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        region = RNG.choice(REGIONS)
        # Hire date spread across 2020–2024
        year = RNG.randint(2020, 2024)
        month = RNG.randint(1, 12)
        day = RNG.randint(1, 28)
        hire_date = f"{year}-{month:02d}-{day:02d}"
        # Salary by department
        bases = {
            "Engineering": (110_000, 165_000),
            "Sales": (75_000, 130_000),
            "Customer Success": (70_000, 110_000),
            "HR": (65_000, 95_000),
            "Finance": (80_000, 120_000),
        }
        lo, hi = bases[dept]
        salary = round(RNG.uniform(lo, hi), -2)  # round to nearest 100
        satisfaction = round(RNG.uniform(3.5, 9.8), 1)
        status = "terminated" if RNG.random() < 0.10 else "active"
        rows.append((
            _rand_name(),
            dept,
            region,
            hire_date,
            salary,
            satisfaction,
            status,
        ))
    return rows


DDL = """
CREATE TABLE IF NOT EXISTS customers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    region          TEXT    NOT NULL,
    tier            TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    signup_date     TEXT    NOT NULL,
    mrr             REAL    NOT NULL,
    industry        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sales (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    amount          REAL    NOT NULL,
    sale_date       TEXT    NOT NULL,
    region          TEXT    NOT NULL,
    rep_name        TEXT    NOT NULL,
    product         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS employees (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    department      TEXT    NOT NULL,
    region          TEXT    NOT NULL,
    hire_date       TEXT    NOT NULL,
    annual_salary   REAL    NOT NULL,
    satisfaction_score REAL NOT NULL,
    status          TEXT    NOT NULL
);
"""


def main():
    parser = argparse.ArgumentParser(description="Seed the SQLite database")
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to output SQLite file (default: data/database.sqlite)",
    )
    args = parser.parse_args()
    db_path = args.db

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Drop existing tables for idempotency
    cur.executescript("DROP TABLE IF EXISTS sales; DROP TABLE IF EXISTS customers; DROP TABLE IF EXISTS employees;")
    cur.executescript(DDL)

    customers = build_customers()
    cur.executemany(
        "INSERT INTO customers (name, region, tier, status, signup_date, mrr, industry) VALUES (?,?,?,?,?,?,?)",
        customers,
    )

    sales = build_sales(customers)
    cur.executemany(
        "INSERT INTO sales (customer_id, amount, sale_date, region, rep_name, product) VALUES (?,?,?,?,?,?)",
        sales,
    )

    employees = build_employees()
    cur.executemany(
        "INSERT INTO employees (name, department, region, hire_date, annual_salary, satisfaction_score, status) VALUES (?,?,?,?,?,?,?)",
        employees,
    )

    con.commit()
    con.close()

    print(f"Seeded {db_path}")
    print(f"  customers : {len(customers)}")
    print(f"  sales     : {len(sales)}")
    print(f"  employees : {len(employees)}")


if __name__ == "__main__":
    main()
