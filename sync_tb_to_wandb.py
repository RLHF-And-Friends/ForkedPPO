#!/usr/bin/env python3.9

import argparse
import os
import subprocess
import glob
import time


def find_offline_wandb_runs(wandb_dir):
    """
    Находит все оффлайн-папки wandb в указанной директории.
    
    Args:
        wandb_dir: Путь к директории с оффлайн-запусками wandb
        
    Returns:
        Список путей к оффлайн-папкам
    """
    if not os.path.exists(wandb_dir):
        print(f"Ошибка: Директория {wandb_dir} не существует")
        return []
    
    offline_runs = []
    
    # Поиск папок с форматом "offline-run-*"
    for item in os.listdir(wandb_dir):
        item_path = os.path.join(wandb_dir, item)
        if os.path.isdir(item_path) and item.startswith("offline-run-"):
            offline_runs.append(item_path)
    
    return offline_runs


def sync_with_wandb_cli(run_path):
    """
    Синхронизирует оффлайн-запуск wandb с сервером, используя CLI команду wandb sync.
    
    Args:
        run_path: Путь к директории с оффлайн-запуском
        
    Returns:
        Успешно ли выполнена синхронизация
    """
    try:
        print(f"Запуск команды: wandb sync {run_path}")
        result = subprocess.run(['wandb', 'sync', run_path], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               check=True)
        
        print("Вывод команды:")
        print(result.stdout)
        
        if result.returncode == 0:
            print("Синхронизация успешно завершена!")
            return True
        else:
            print(f"Ошибка при выполнении команды wandb sync: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении команды wandb sync: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return False


def sync_all_offline_runs_cli(wandb_dir, wait_between_syncs=2):
    """
    Синхронизирует все оффлайн-запуски wandb с сервером, используя CLI команду wandb sync.
    
    Args:
        wandb_dir: Путь к директории с оффлайн-запусками wandb
        wait_between_syncs: Время ожидания между синхронизациями в секундах
        
    Returns:
        Количество успешно синхронизированных запусков
    """
    offline_runs = find_offline_wandb_runs(wandb_dir)
    
    if not offline_runs:
        print(f"В директории {wandb_dir} не найдены оффлайн-запуски wandb")
        return 0
    
    print(f"Найдено {len(offline_runs)} оффлайн-запусков wandb:")
    for i, run_path in enumerate(offline_runs):
        print(f"{i+1}. {os.path.basename(run_path)}")
    
    success_count = 0
    
    for i, run_path in enumerate(offline_runs):
        print(f"\n[{i+1}/{len(offline_runs)}] Обработка: {os.path.basename(run_path)}")
        
        # Попытка синхронизации через CLI
        if sync_with_wandb_cli(run_path):
            success_count += 1
        
        # Задержка между синхронизациями
        if i < len(offline_runs) - 1:
            print(f"Ожидание {wait_between_syncs} секунд перед следующей синхронизацией...")
            time.sleep(wait_between_syncs)
    
    print(f"\nИтого: успешно синхронизировано {success_count} из {len(offline_runs)} запусков")
    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Синхронизация оффлайн-запусков wandb с Weights & Biases, используя CLI команду")
    
    parser.add_argument(
        "--wandb-dir",
        type=str,
        required=True,
        help="Путь к директории с оффлайн-запусками wandb"
    )
    
    parser.add_argument(
        "--wait",
        type=int,
        default=2,
        help="Время ожидания в секундах между синхронизациями (по умолчанию: 2)"
    )
    
    args = parser.parse_args()
    
    sync_all_offline_runs_cli(args.wandb_dir, args.wait) 