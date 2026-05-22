from pathlib import Path

# Путь к папке с файлами
folder = Path(r"dataset/images/val_degraded_dataset/val4/")
print(folder)
# Перебираем все txt-файлы
for file in folder.glob("*.*"):
    print(file)
    # Получаем имя без расширения
    old_name = file.stem
    new_name = f"smoke_{old_name}.txt"
    new_path = file.with_name(new_name)
    # Переименование
    file.rename(new_path)
    print(f"{file.name} -> {new_name}")
print("Готово.")