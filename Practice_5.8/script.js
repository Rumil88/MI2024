document.addEventListener('DOMContentLoaded', function() {
    // Кнопки перемикання режимів
    const simpleButton = document.getElementById('simple-mode');
    const detailedButton = document.getElementById('detailed-mode');
    const detailedInfo = document.getElementById('detailed-info');
    
    // Контекст для графіка
    const ctx = document.getElementById('droneChart').getContext('2d');
    let droneChart;
    
    // Дані про застосування FPV-дронів (на основі відкритих джерел)
    // Примітка: ці дані є приблизними і базуються на відкритих джерелах
    const droneData = {
        // Спрощені дані (по роках)
        simple: {
            labels: ['2022', '2023', '2024'],
            ukraine: [1200, 5800, 8500],
            russia: [800, 3200, 5100]
        },
        // Детальні дані (по кварталах)
        detailed: {
            labels: [
                'Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022',
                'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023',
                'Q1 2024', 'Q2 2024'
            ],
            ukraine: [200, 300, 350, 350, 900, 1400, 1600, 1900, 2800, 5700],
            russia: [150, 200, 220, 230, 500, 700, 900, 1100, 1600, 3500]
        }
    };
    
    // Функція для створення графіка
    function createChart(mode) {
        // Якщо графік вже існує, знищуємо його
        if (droneChart) {
            droneChart.destroy();
        }
        
        // Вибираємо дані відповідно до режиму
        const data = droneData[mode];
        
        // Створюємо новий графік
        droneChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'ЗСУ',
                        data: data.ukraine,
                        backgroundColor: '#3498db',
                        borderColor: '#2980b9',
                        borderWidth: 1
                    },
                    {
                        label: 'РФ',
                        data: data.russia,
                        backgroundColor: '#e74c3c',
                        borderColor: '#c0392b',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: mode === 'simple' ? 
                            'Кількість застосованих FPV-дронів (по роках)' : 
                            'Кількість застосованих FPV-дронів (по кварталах)',
                        font: {
                            size: 16
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.raw + ' одиниць';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Кількість дронів'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: mode === 'simple' ? 'Рік' : 'Квартал'
                        }
                    }
                }
            }
        });
    }
    
    // Обробники подій для кнопок
    simpleButton.addEventListener('click', function() {
        simpleButton.classList.add('active');
        detailedButton.classList.remove('active');
        detailedInfo.classList.add('hidden');
        createChart('simple');
    });
    
    detailedButton.addEventListener('click', function() {
        detailedButton.classList.add('active');
        simpleButton.classList.remove('active');
        detailedInfo.classList.remove('hidden');
        createChart('detailed');
    });
    
    // Ініціалізація графіка в спрощеному режимі
    createChart('simple');
});