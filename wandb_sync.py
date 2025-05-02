#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import concurrent.futures
import threading

# Создаем замок для предотвращения перемешивания вывода
print_lock = threading.Lock()

def sync_single_folder(path, index, total):
    """
    Синхронизирует один путь wandb
    """
    with print_lock:
        print(f"[{index}/{total}] Запуск синхронизации {path}...")
    
    if not os.path.exists(path):
        with print_lock:
            print(f"Ошибка: путь {path} не существует")
        return path, False, "Путь не существует"
        
    try:        
        result = subprocess.run(["wandb", "sync", path], check=True)
        
        return path, True, None
    except subprocess.CalledProcessError as e:
        with print_lock:
            print(f"Ошибка при синхронизации {path}: {e}")
        return path, False, str(e)

def sync_wandb_folders(paths, max_workers=None):
    """
    Запускает wandb sync для каждого из указанных путей параллельно
    """
    successful = []
    failed = []
    
    # Используем ThreadPoolExecutor для параллельного запуска
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Создаем список задач
        future_to_path = {
            executor.submit(sync_single_folder, path, i+1, len(paths)): path 
            for i, path in enumerate(paths)
        }
        
        # Обрабатываем результаты по мере их завершения
        for future in concurrent.futures.as_completed(future_to_path):
            path, success, error = future.result()
            if success:
                successful.append(path)
            else:
                failed.append((path, error))
    
    # Вывод итогов
    print("\n--- Итоги синхронизации ---")
    print(f"Успешно синхронизировано: {len(successful)}/{len(paths)}")
    
    if failed:
        print(f"Не удалось синхронизировать: {len(failed)}/{len(paths)}")
        print("Список неудачных синхронизаций:")
        for path, error in failed:
            print(f"- {path}: {error}")
    
    return successful, failed

def main():
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(
        description="Параллельная синхронизация оффлайн-вычислений wandb"
    )
    parser.add_argument(
        "paths", 
        nargs="+", 
        help="Пути до оффлайн-вычислений wandb (можно указать несколько)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Максимальное количество параллельных процессов (по умолчанию: количество CPU)"
    )
    
    args = parser.parse_args()
    
    # Запускаем синхронизацию
    successful, failed = sync_wandb_folders(args.paths, max_workers=args.workers)
    
    # Если были ошибки, выходим с ненулевым кодом
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
