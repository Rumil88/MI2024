# Імпорт необхідних бібліотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Шлях до файлу даних
file_path = 'Mobiles Dataset (2025).csv'

# Завантаження даних
print('Завантаження даних...')
# Спробуємо різні кодування для вирішення проблеми з читанням файлу
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        try:
            df = pd.read_csv(file_path, encoding='cp1252')
        except Exception as e:
            print(f"Помилка при читанні файлу: {e}")
            exit(1)

# Перегляд перших рядків
print('\nПерші 5 рядків даних:')
print(df.head())

# Огляд інформації про набір даних
print('\nІнформація про набір даних:')
print(df.info())

# Перевірка на пропущені значення
print('\nКількість пропущених значень у кожному стовпці:')
print(df.isnull().sum())

# Очищення даних
print('\nОчищення даних...')

# Функція для безпечного перетворення стрічок на числа
def safe_convert_to_float(series):
    # Спочатку замінюємо порожні стрічки на NaN
    series = series.replace('', np.nan)
    # Повертаємо серію, якщо вона вже числова
    if pd.api.types.is_numeric_dtype(series):
        return series
    # Інакше намагаємося перетворити
    try:
        return pd.to_numeric(series, errors='coerce')
    except:
        return series

# Перетворення цінових стовпців на числові значення (видалення символів валют)
price_columns = [col for col in df.columns if 'Price' in col]
for col in price_columns:
    # Видаляємо всі нечислові символи, крім крапки
    df[col] = df[col].astype(str).replace('[^\d.]', '', regex=True)
    # Безпечно перетворюємо на числа
    df[col] = safe_convert_to_float(df[col])

# Перетворення стовпця ваги на числовий (видалення 'g')
if 'Mobile Weight' in df.columns:
    df['Mobile Weight'] = df['Mobile Weight'].astype(str).str.replace('g', '')
    df['Mobile Weight'] = safe_convert_to_float(df['Mobile Weight'])

# Перетворення стовпця розміру екрану на числовий (видалення 'inches')
if 'Screen Size' in df.columns:
    df['Screen Size'] = df['Screen Size'].astype(str).str.replace(' inches', '')
    df['Screen Size'] = safe_convert_to_float(df['Screen Size'])

# Перетворення стовпця ємності батареї на числовий (видалення 'mAh')
if 'Battery Capacity' in df.columns:
    df['Battery Capacity'] = df['Battery Capacity'].astype(str).str.replace('mAh', '').str.replace(',', '')
    df['Battery Capacity'] = safe_convert_to_float(df['Battery Capacity'])

# Перетворення стовпців камер на числові (беремо тільки перше число)
def extract_first_number(text):
    import re
    if pd.isna(text) or text == '':
        return np.nan
    match = re.search(r'(\d+)', str(text))
    if match:
        return int(match.group(1))
    return np.nan

if 'Front Camera' in df.columns:
    df['Front Camera'] = df['Front Camera'].apply(extract_first_number)
if 'Back Camera' in df.columns:
    df['Back Camera'] = df['Back Camera'].apply(extract_first_number)

# Заповнення пропущених значень середнім значенням для числових стовпців
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df[col].isna().any():
        mean_value = df[col].mean()
        if not pd.isna(mean_value):
            df[col] = df[col].fillna(mean_value)

# Заповнення пропущених значень для категоріальних стовпців найчастішим значенням
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df[col].isna().any() and len(df[col].dropna()) > 0:
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value[0])

# Видалення дублікатів
df.drop_duplicates(inplace=True)

print('\nПісля очищення даних:')
print(df.head())

# Трансформація даних
print('\nТрансформація даних...')

# Нормалізація числових змінних
scaler = MinMaxScaler()
# Перевіряємо, які стовпці є в даних і є числовими
numeric_cols_to_scale = [col for col in ['Mobile Weight', 'RAM', 'Front Camera', 'Back Camera', 
                         'Battery Capacity', 'Screen Size'] + price_columns 
                         if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

# Перевіряємо, чи є дані для нормалізації
if numeric_cols_to_scale:
    # Створюємо копію даних для нормалізації
    df_scaled = df.copy()
    # Замінюємо NaN на середні значення перед нормалізацією
    df_temp = df[numeric_cols_to_scale].fillna(df[numeric_cols_to_scale].mean())
    # Нормалізуємо дані
    df_scaled[numeric_cols_to_scale] = scaler.fit_transform(df_temp)
else:
    df_scaled = df.copy()

# One-Hot Encoding для категоріальних змінних
df_encoded = pd.get_dummies(df, columns=['Company Name', 'Processor'])

print('\nПісля трансформації даних (перші 5 рядків):')
print(df_encoded.head())

# Декомпозиція набору даних
print('\nДекомпозиція набору даних...')

# Вибір цільової змінної (наприклад, ціна в США)
target_column = 'Launched Price (USA)'

# Перевіряємо, чи є цільова змінна в даних
if target_column not in df_encoded.columns:
    print(f"Помилка: Цільова змінна '{target_column}' відсутня в даних.")
    # Вибираємо іншу цінову змінну, якщо можливо
    price_cols = [col for col in df_encoded.columns if 'Price' in col]
    if price_cols:
        target_column = price_cols[0]
        print(f"Використовуємо '{target_column}' як цільову змінну.")
    else:
        print("Немає цінових змінних для аналізу. Завершення програми.")
        exit(1)

# Вибір ознак (виключаємо інші цінові стовпці та цільову змінну)
feature_columns = [col for col in df_encoded.columns 
                  if ('Price' not in col or col == target_column) and col != target_column]

X = df_encoded[feature_columns]
y = df_encoded[target_column]

# Поділ на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Розмір навчальної вибірки: {X_train.shape}')
print(f'Розмір тестової вибірки: {X_test.shape}')

# Кореляційна матриця для числових змінних
correlation_matrix = df[numeric_columns].corr()
print('\nКореляційна матриця:')
print(correlation_matrix)

# Описова статистика та візуалізація
print('\nОписова статистика:')
print(df[numeric_columns].describe())

# Створення папки для збереження графіків, якщо вона не існує
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Візуалізація даних
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Кореляційна матриця числових змінних')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

# Гістограма для ціни в США
plt.figure(figsize=(10, 6))
sns.histplot(df['Launched Price (USA)'], bins=20, kde=True)
plt.title('Розподіл цін на мобільні телефони в США')
plt.xlabel('Ціна (USD)')
plt.ylabel('Кількість')
plt.tight_layout()
plt.savefig('plots/price_distribution.png')
plt.close()

# Коробкова діаграма для ємності батареї за компаніями
plt.figure(figsize=(12, 8))
sns.boxplot(x='Company Name', y='Battery Capacity', data=df)
plt.title('Ємність батареї за виробниками')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('plots/battery_by_company.png')
plt.close()

# Діаграма розсіювання: ємність батареї vs ціна
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Battery Capacity', y='Launched Price (USA)', hue='Company Name', data=df)
plt.title('Залежність ціни від ємності батареї')
plt.xlabel('Ємність батареї (mAh)')
plt.ylabel('Ціна (USD)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plots/price_vs_battery.png')
plt.close()

# Побудова моделі
print('\nПобудова моделі лінійної регресії...')

# Створення та навчання моделі лінійної регресії
model = LinearRegression()
model.fit(X_train, y_train)

# Прогноз на тестовій вибірці
y_pred = model.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Середньоквадратична помилка (MSE): {mse:.2f}')
print(f'Коефіцієнт детермінації (R²): {r2:.2f}')

# Візуалізація результатів моделі
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Фактична ціна')
plt.ylabel('Прогнозована ціна')
plt.title('Порівняння фактичних та прогнозованих цін')
plt.tight_layout()
plt.savefig('plots/model_prediction.png')
plt.close()

# Аналіз важливості ознак
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(model.coef_)})
feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Топ-10 найважливіших ознак для прогнозування ціни')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

print('\nАналіз завершено. Графіки збережено в папці "plots".')