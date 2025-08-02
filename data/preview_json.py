# -*- coding: utf-8 -*-
# 编码格式: UTF-8
# 文件用途: 用于预览json文件的前十行

import json

def preview_json(json_path):
    try:
        with open(json_path, 'r') as file:
            # 读取前10行
            data = []
            for _ in range(10):
                line = file.readline().strip()
                if not line:
                    break
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line: {e}")
                    continue
            
            # 先打印原始数据看看格式
            print("Raw data format:")
            print(data[0])
            print("\n")
            
            # 打印前10条数据
            print("First 10 items:")
            for i, item in enumerate(data):
                try:
                    print(f"\n{i+1}. Raw item:")
                    print(json.dumps(item, indent=2))
                    print("-" * 100)
                except Exception as e:
                    print(f"Error processing item {i+1}: {str(e)}")
                    continue

    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    preview_json("/home/zjc/CoIn/data/tacred/data_merged.json")
