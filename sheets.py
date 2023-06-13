import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime


# Подключение к Google Таблицам
def connect_to_google_sheets():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "credentials.json", scope
    )
    client = gspread.authorize(credentials)
    return client


def check_time_and_get_data():
    # Подключение к Google Таблицам
    client = connect_to_google_sheets()

    # Открытие таблицы
    sheet = client.open_by_url(
        "https://docs.google.com/spreadsheets/d/192KHoUmcwzeNq8sCHrpf8bKDK2lHXmkc5YdQ7dBhqto/edit?usp=sharing"
    ).sheet1

    # Получение текущего времени и даты
    current_time = datetime.datetime.now().time()

    # Поиск времени в первом столбце таблицы
    time_column = sheet.col_values(1)
    data_column = sheet.col_values(2)

    # Проверка соответствия времени диапазону
    for i in range(len(time_column)):
        time_range = time_column[i]
        start_time, end_time = time_range.split("-")
        start_time = datetime.datetime.strptime(start_time.strip(), "%H:%M").time()
        end_time = datetime.datetime.strptime(end_time.strip(), "%H:%M").time()

        if start_time <= current_time <= end_time:
            data = data_column[i]
            return data

    return None


# Обновление посещаемости в таблице group
def update_attendance(student_name, sheet_name):
    student_name = student_name.split('.')[0]
    # Подключение к Google Таблицам
    client = connect_to_google_sheets()

    # Открытие таблицы group
    sheet = client.open_by_url(
        "https://docs.google.com/spreadsheets/d/1TWZsOJTJXKMBLQZxCwaYMQqG08fObvFuR88wY-D0n9c/edit?usp=sharing"
    ).worksheet(
        sheet_name
    )  # Замените на имя листа с нужным предметом

    # Получение текущей даты
    current_date = datetime.datetime.now().strftime("%d.%m")

    # Поиск студентов в первом столбце таблицы
    student_column = sheet.col_values(1)
    student_row_index = [
        i for i, student in enumerate(student_column) if student == student_name
    ][0]

    # Поиск даты в первой строке таблицы
    date_row = sheet.row_values(1)
    date_column_index = [i for i, date in enumerate(date_row) if date == current_date][
        0
    ]

    # Получение значений столбца с датой
    date_column_row = sheet.col_values(date_column_index + 1)

    # Установка значений
    for row in range(2, len(student_column) + 1):
        if row == student_row_index + 1:
            sheet.update_cell(row, date_column_index + 1, "1")
            continue

        date_column_row_value = None
        try:
            date_column_row_value = date_column_row[row - 1]
        except IndexError:
            pass

        if date_column_row_value != "1":
            sheet.update_cell(row, date_column_index + 1, "0")


