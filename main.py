#!/venv/Scripts/python.exe

import os
from recognition import load_image_from_camera
from recognition import recognize_face
from sheets import check_time_and_get_data
from sheets import update_attendance

print("Запускаем камеру...")
image = load_image_from_camera()
print("Распознаем лицо...")
result = recognize_face(image)


if result is not None:
    data = check_time_and_get_data()
    student_names = os.listdir("photos")
    student_name = student_names[result]
    update_attendance(student_name, data)
    print("Распознано лицо студента:", student_name)
else:
    print("Лицо не найдено")


