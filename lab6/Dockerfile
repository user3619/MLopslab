# Dockerfile
# базовый образ
FROM python:3.10-slim

# рабочая директория проекта
WORKDIR /fast_app

# копируем файлы
COPY . .

# устанавливаем зависимости
RUN pip3 install --no-cache-dir -r ./requirements.txt

# порт для взаимодействия с контейнером
EXPOSE 8005

# запуск приложения
CMD ["python", "./main.py"]