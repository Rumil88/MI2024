import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Функція для безпечного перетворення стовпців
def safe_convert_column(df, column_name, pattern=None, replace_dict=None, extract_pattern=None):
    try:
        # Створюємо копію стовпця для обробки
        series = df[column_name].astype(str)
        
        # Заміна значень за словником
        if replace_dict:
            for old, new in replace_dict.items():
                series = series.str.replace(old, new)
        
        # Видалення патерну за регулярним виразом
        if pattern:
            series = series.str.replace(pattern, '', regex=True)
        
        # Вилучення значень за регулярним виразом
        if extract_pattern:
            extracted = series.str.extract(extract_pattern)
            if extracted is not None and not extracted.empty:
                series = extracted[0]
        
        # Перетворення на числовий тип
        return pd.to_numeric(series, errors='coerce')
    except Exception as e:
        print(f"Помилка при обробці стовпця {column_name}: {e}")
        return pd.Series([np.nan] * len(df))

# Завантаження даних
print("Завантаження даних...")
try:
    # Спробуємо різні кодування
    encodings = ['latin1', 'cp1252', 'ISO-8859-1', 'utf-8-sig']
    for encoding in encodings:
        try:
            df = pd.read_csv('Mobiles Dataset (2025).csv', encoding=encoding)
            print(f"Файл успішно відкрито з кодуванням {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Помилка при спробі відкрити файл з кодуванням {encoding}: {e}")
    else:
        raise Exception("Не вдалося відкрити файл з жодним із кодувань")
    
    # Перегляд перших рядків
    print("\nПерші 5 рядків даних:")
    print(df.head())
    
    # Огляд інформації про набір даних
    print("\nІнформація про набір даних:")
    print(df.info())
    
    # Перевірка на пропущені значення
    print("\nКількість пропущених значень у кожному стовпці:")
    print(df.isnull().sum())
    
    # Перевірка на дублікати
    duplicates = df.duplicated().sum()
    print(f"\nКількість дублікатів: {duplicates}")
    
    # Очищення даних
    print("\nОчищення даних...")
    
    # Заміна 'Not available' на NaN для всіх стовпців
    df = df.replace('Not available', np.nan)
    
    # Видалення дублікатів
    df.drop_duplicates(inplace=True)
    
    # Перетворення категоріальних змінних
    numeric_columns = []
    
    # Обробка стовпця ваги мобільного пристрою
    if 'Mobile Weight' in df.columns:
        df['Weight'] = safe_convert_column(df, 'Mobile Weight', pattern='[^\d.]', extract_pattern='(\d+)')
        numeric_columns.append('Weight')
    
    # Обробка стовпця RAM
    if 'RAM' in df.columns:
        df['RAM_GB'] = safe_convert_column(df, 'RAM', pattern='GB|g|,|\s')
        numeric_columns.append('RAM_GB')
    
    # Обробка стовпця розміру екрану
    if 'Screen Size' in df.columns:
        df['Screen_Size_Inches'] = safe_convert_column(df, 'Screen Size', pattern='inches|"|\'|\s')
        numeric_columns.append('Screen_Size_Inches')
    
    # Обробка стовпця з ємністю батареї
    if 'Battery Capacity' in df.columns:
        df['Battery_mAh'] = safe_convert_column(df, 'Battery Capacity', pattern='mAh|\s', extract_pattern='(\d+,?\d*)')
        numeric_columns.append('Battery_mAh')
    
    # Обробка стовпців з цінами
    price_columns = [col for col in df.columns if 'Price' in col]
    for col in price_columns:
        clean_name = col.replace('Launched Price ', '').replace('(', '').replace(')', '')
        new_col = f"{clean_name}_Price"
        replace_dict = {'PKR': '', 'INR': '', 'CNY': '', 'USD': '', 'AED': '', ',': '', ' ': ''}
        df[new_col] = safe_convert_column(df, col, replace_dict=replace_dict)
        numeric_columns.append(new_col)
    
    # Заповнення пропущених значень середнім для числових стовпців
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    
    # Описова статистика для числових змінних
    print("\nОписова статистика для числових змінних:")
    print(df[numeric_columns].describe())
    
    # Візуалізація даних
    print("\nСтворення візуалізацій...")
    
    # Налаштування розміру графіків
    plt.figure(figsize=(15, 10))
    
    # Гістограма для ваги мобільних пристроїв
    plt.subplot(2, 3, 1)
    if 'Weight' in df.columns:
        sns.histplot(df['Weight'].dropna(), bins=20, kde=True)
        plt.title('Розподіл ваги мобільних пристроїв')
        plt.xlabel('Вага (г)')
        plt.ylabel('Кількість')
    
    # Коробкова діаграма для цін в USD
    plt.subplot(2, 3, 2)
    if 'USA_Price' in df.columns:
        sns.boxplot(y=df['USA_Price'].dropna())
        plt.title('Розподіл цін мобільних пристроїв (USD)')
        plt.ylabel('Ціна (USD)')
    
    # Діаграма розсіювання для ваги та ціни
    plt.subplot(2, 3, 3)
    if 'Weight' in df.columns and 'USA_Price' in df.columns:
        sns.scatterplot(x=df['Weight'].dropna(), y=df['USA_Price'].dropna())
        plt.title('Залежність між вагою та ціною')
        plt.xlabel('Вага (г)')
        plt.ylabel('Ціна (USD)')
    
    # Діаграма розсіювання для RAM та ціни
    plt.subplot(2, 3, 4)
    if 'RAM_GB' in df.columns and 'USA_Price' in df.columns:
        sns.scatterplot(x=df['RAM_GB'].dropna(), y=df['USA_Price'].dropna())
        plt.title('Залежність між RAM та ціною')
        plt.xlabel('RAM (GB)')
        plt.ylabel('Ціна (USD)')
    
    # Діаграма розсіювання для ємності батареї та ціни
    plt.subplot(2, 3, 5)
    if 'Battery_mAh' in df.columns and 'USA_Price' in df.columns:
        sns.scatterplot(x=df['Battery_mAh'].dropna(), y=df['USA_Price'].dropna())
        plt.title('Залежність між ємністю батареї та ціною')
        plt.xlabel('Ємність батареї (mAh)')
        plt.ylabel('Ціна (USD)')
    
    # Діаграма розсіювання для розміру екрану та ціни
    plt.subplot(2, 3, 6)
    if 'Screen_Size_Inches' in df.columns and 'USA_Price' in df.columns:
        sns.scatterplot(x=df['Screen_Size_Inches'].dropna(), y=df['USA_Price'].dropna())
        plt.title('Залежність між розміром екрану та ціною')
        plt.xlabel('Розмір екрану (дюйми)')
        plt.ylabel('Ціна (USD)')
    
    plt.tight_layout()
    plt.savefig('mobile_visualizations.png')
    
    # Кореляційний аналіз для числових змінних
    print("\nКореляційний аналіз...")
    correlation_matrix = df[numeric_columns].corr()
    
    # Візуалізація кореляційної матриці
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Кореляційна матриця')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    print("\nАналіз завершено. Графіки збережено у файлах 'mobile_visualizations.png' та 'correlation_matrix.png'")

except Exception as e:
    print(f"Виникла помилка при виконанні аналізу: {e}")