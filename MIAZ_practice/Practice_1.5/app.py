from flask import Flask, request, jsonify, render_template
import psycopg2
from datetime import datetime

app = Flask(__name__)

# Функція підключення до бази даних
def get_db_connection():
    return psycopg2.connect(
        host='localhost',
        database='students',
        user='postgres',
        password='admin'
    )

# Створення таблиці students при першому запуску
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            enrollment_date DATE NOT NULL
        );
    ''')
    conn.commit()
    cur.close()
    conn.close()

# Маршрут для головної сторінки
@app.route('/')
def index():
    return render_template('index.html')

# GET /api/students: отримати всіх студентів
@app.route('/api/students', methods=['GET'])
def get_students():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM students;')
    students = [{
        'student_id': row[0],
        'name': row[1],
        'enrollment_date': row[2].strftime('%Y-%m-%d')
    } for row in cur.fetchall()]
    cur.close()
    conn.close()
    return jsonify(students)

# POST /api/students: додати нового студента
@app.route('/api/students', methods=['POST'])
def add_student():
    data = request.get_json()
    name = data.get('name')
    enrollment_date = data.get('enrollment_date')
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO students (name, enrollment_date) VALUES (%s, %s) RETURNING student_id;',
        (name, enrollment_date)
    )
    student_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    
    return jsonify({'student_id': student_id, 'message': 'Студента успішно додано'}), 201

# DELETE /api/students/<student_id>: видалити студента
@app.route('/api/students/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM students WHERE student_id = %s;', (student_id,))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'message': 'Студента успішно видалено'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
