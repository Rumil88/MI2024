import sqlite3
import pandas as pd
import os
import csv
from datetime import datetime

# Шлях до бази даних
DB_PATH = 'logistics_database.db'

# Шляхи до CSV файлів
LOGISTICS_UNITS_CSV = 'logistics_units.csv'
LOGISTICS_PERSONNEL_CSV = 'logistics_personnel.csv'

# Директорія для збереження результатів запитів
RESULTS_DIR = 'query_results'

def create_connection():
    """Створює з'єднання з базою даних SQLite"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        print(f"Успішне підключення до бази даних {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        print(f"Помилка при підключенні до бази даних: {e}")
    return conn

def create_tables(conn):
    """Створює необхідні таблиці в базі даних"""
    try:
        cursor = conn.cursor()
        
        # Створення таблиці Logistics_Units
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Logistics_Units (
            Logistics_Unit_ID INTEGER PRIMARY KEY,
            Logistics_Unit_Name TEXT NOT NULL,
            Supplies_Available INTEGER,
            Base_Location TEXT
        )
        ''')
        
        # Створення таблиці Logistics_Personnel
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Logistics_Personnel (
            ID INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Position TEXT,
            Logistics_Unit_ID INTEGER,
            Date_Assigned TEXT,
            FOREIGN KEY (Logistics_Unit_ID) REFERENCES Logistics_Units (Logistics_Unit_ID)
        )
        ''')
        
        conn.commit()
        print("Таблиці успішно створені")
    except sqlite3.Error as e:
        print(f"Помилка при створенні таблиць: {e}")

def import_data_from_csv(conn):
    """Імпортує дані з CSV файлів у таблиці бази даних"""
    try:
        # Імпорт даних з logistics_units.csv
        units_df = pd.read_csv(LOGISTICS_UNITS_CSV)
        units_df.to_sql('Logistics_Units', conn, if_exists='replace', index=False)
        
        # Імпорт даних з logistics_personnel.csv
        personnel_df = pd.read_csv(LOGISTICS_PERSONNEL_CSV)
        personnel_df.to_sql('Logistics_Personnel', conn, if_exists='replace', index=False)
        
        print("Дані успішно імпортовані з CSV файлів")
    except Exception as e:
        print(f"Помилка при імпорті даних: {e}")

def ensure_results_directory():
    """Перевіряє наявність директорії для результатів запитів і створює її, якщо потрібно"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Створено директорію {RESULTS_DIR} для збереження результатів запитів")

def execute_query(conn, query, filename, description):
    """Виконує SQL запит і зберігає результат у CSV файл"""
    try:
        # Виконання запиту
        result_df = pd.read_sql_query(query, conn)
        
        # Збереження результату в CSV
        ensure_results_directory()
        file_path = os.path.join(RESULTS_DIR, filename)
        result_df.to_csv(file_path, index=False, encoding='utf-8')
        
        print(f"Запит '{description}' виконано успішно. Результат збережено у {file_path}")
        return result_df
    except Exception as e:
        print(f"Помилка при виконанні запиту '{description}': {e}")
        return None

def run_all_queries(conn):
    """Виконує всі підготовлені SQL запити"""
    # 1. Вибірка всіх логістичних підрозділів
    query1 = "SELECT * FROM Logistics_Units"
    execute_query(conn, query1, "all_logistics_units.csv", "Всі логістичні підрозділи")
    
    # 2. Вибірка всього персоналу
    query2 = "SELECT * FROM Logistics_Personnel"
    execute_query(conn, query2, "all_personnel.csv", "Весь персонал")
    
    # 3. Підрахунок кількості працівників у кожному підрозділі
    query3 = """
    SELECT lu.Logistics_Unit_Name, COUNT(lp.ID) as Employee_Count
    FROM Logistics_Units lu
    LEFT JOIN Logistics_Personnel lp ON lu.Logistics_Unit_ID = lp.Logistics_Unit_ID
    GROUP BY lu.Logistics_Unit_ID
    ORDER BY Employee_Count DESC
    """
    execute_query(conn, query3, "employees_per_unit.csv", "Кількість працівників у кожному підрозділі")
    
    # 4. Вибірка працівників з посадою 'Водій'
    query4 = "SELECT * FROM Logistics_Personnel WHERE Position = 'Водій'"
    execute_query(conn, query4, "drivers.csv", "Працівники з посадою 'Водій'")
    
    # 5. Вибірка підрозділів з кількістю запасів більше 1500
    query5 = "SELECT * FROM Logistics_Units WHERE Supplies_Available > 1500"
    execute_query(conn, query5, "high_supply_units.csv", "Підрозділи з запасами > 1500")
    
    # 6. Вибірка працівників, призначених після 1 березня 2022
    query6 = "SELECT * FROM Logistics_Personnel WHERE Date_Assigned > '2022-03-01'"
    execute_query(conn, query6, "recent_employees.csv", "Працівники, призначені після 01.03.2022")
    
    # 7. Об'єднання таблиць для отримання інформації про працівників з назвами підрозділів
    query7 = """
    SELECT lp.ID, lp.Name, lp.Position, lu.Logistics_Unit_Name, lu.Base_Location
    FROM Logistics_Personnel lp
    JOIN Logistics_Units lu ON lp.Logistics_Unit_ID = lu.Logistics_Unit_ID
    """
    execute_query(conn, query7, "employees_with_units.csv", "Працівники з назвами підрозділів")
    
    # 8. Підрахунок кількості працівників за посадами
    query8 = """
    SELECT Position, COUNT(*) as Count
    FROM Logistics_Personnel
    GROUP BY Position
    ORDER BY Count DESC
    """
    execute_query(conn, query8, "positions_count.csv", "Кількість працівників за посадами")
    
    # 9. Вибірка підрозділів з найбільшою кількістю запасів
    query9 = "SELECT * FROM Logistics_Units ORDER BY Supplies_Available DESC LIMIT 3"
    execute_query(conn, query9, "top_supply_units.csv", "Підрозділи з найбільшою кількістю запасів")
    
    # 10. Вибірка працівників з Києва
    query10 = """
    SELECT lp.ID, lp.Name, lp.Position, lu.Base_Location
    FROM Logistics_Personnel lp
    JOIN Logistics_Units lu ON lp.Logistics_Unit_ID = lu.Logistics_Unit_ID
    WHERE lu.Base_Location = 'Київ'
    """
    execute_query(conn, query10, "kyiv_employees.csv", "Працівники з Києва")
    
    # 11. Середня кількість запасів по всіх підрозділах
    query11 = "SELECT AVG(Supplies_Available) as Average_Supplies FROM Logistics_Units"
    execute_query(conn, query11, "average_supplies.csv", "Середня кількість запасів")
    
    # 12. Вибірка працівників, призначених у першій половині 2022 року
    query12 = """
    SELECT * FROM Logistics_Personnel 
    WHERE Date_Assigned BETWEEN '2022-01-01' AND '2022-06-30'
    ORDER BY Date_Assigned
    """
    execute_query(conn, query12, "first_half_2022_employees.csv", "Працівники, призначені у першій половині 2022")
    
    # 13. Кількість працівників за місяцями призначення
    query13 = """
    SELECT strftime('%Y-%m', Date_Assigned) as Month, COUNT(*) as Employee_Count
    FROM Logistics_Personnel
    GROUP BY Month
    ORDER BY Month
    """
    execute_query(conn, query13, "employees_by_month.csv", "Кількість працівників за місяцями призначення")
    
    # 14. Вибірка підрозділів та їх працівників (з використанням підзапиту)
    query14 = """
    SELECT lu.Logistics_Unit_Name, 
           (SELECT COUNT(*) FROM Logistics_Personnel lp WHERE lp.Logistics_Unit_ID = lu.Logistics_Unit_ID) as Employee_Count,
           lu.Supplies_Available,
           lu.Base_Location
    FROM Logistics_Units lu
    """
    execute_query(conn, query14, "units_with_employee_count.csv", "Підрозділи з кількістю працівників")
    
    # 15. Вибірка працівників, які є менеджерами
    query15 = "SELECT * FROM Logistics_Personnel WHERE Position LIKE '%Менеджер%'"
    execute_query(conn, query15, "managers.csv", "Працівники-менеджери")
    
    # 16. Вибірка підрозділів зі східних регіонів
    query16 = "SELECT * FROM Logistics_Units WHERE Base_Location IN ('Харків', 'Дніпро')"
    execute_query(conn, query16, "eastern_units.csv", "Підрозділи зі східних регіонів")
    
    # 17. Вибірка працівників з найбільшим стажем (найраніша дата призначення)
    query17 = "SELECT * FROM Logistics_Personnel ORDER BY Date_Assigned ASC LIMIT 5"
    execute_query(conn, query17, "senior_employees.csv", "Працівники з найбільшим стажем")
    
    # 18. Вибірка підрозділів та їх працівників (з використанням JOIN та GROUP BY)
    query18 = """
    SELECT lu.Logistics_Unit_ID, lu.Logistics_Unit_Name, COUNT(lp.ID) as Employee_Count, 
           GROUP_CONCAT(lp.Name, ', ') as Employee_Names
    FROM Logistics_Units lu
    LEFT JOIN Logistics_Personnel lp ON lu.Logistics_Unit_ID = lp.Logistics_Unit_ID
    GROUP BY lu.Logistics_Unit_ID
    """
    execute_query(conn, query18, "units_with_employee_names.csv", "Підрозділи з іменами працівників")
    
    # 19. Вибірка працівників, які працюють у підрозділах з запасами менше середнього
    query19 = """
    SELECT lp.ID, lp.Name, lp.Position, lu.Logistics_Unit_Name, lu.Supplies_Available
    FROM Logistics_Personnel lp
    JOIN Logistics_Units lu ON lp.Logistics_Unit_ID = lu.Logistics_Unit_ID
    WHERE lu.Supplies_Available < (SELECT AVG(Supplies_Available) FROM Logistics_Units)
    """
    execute_query(conn, query19, "employees_in_low_supply_units.csv", "Працівники у підрозділах з запасами нижче середнього")
    
    # 20. Вибірка підрозділів без працівників (якщо такі є)
    query20 = """
    SELECT lu.*
    FROM Logistics_Units lu
    LEFT JOIN Logistics_Personnel lp ON lu.Logistics_Unit_ID = lp.Logistics_Unit_ID
    WHERE lp.ID IS NULL
    """
    execute_query(conn, query20, "units_without_employees.csv", "Підрозділи без працівників")

def main():
    """Основна функція програми"""
    # Створення з'єднання з базою даних
    conn = create_connection()
    
    if conn is not None:
        # Створення таблиць
        create_tables(conn)
        
        # Імпорт даних з CSV
        import_data_from_csv(conn)
        
        # Виконання всіх запитів
        run_all_queries(conn)
        
        # Закриття з'єднання
        conn.close()
        print("Роботу з базою даних завершено")
    else:
        print("Не вдалося створити з'єднання з базою даних")

if __name__ == "__main__":
    main()