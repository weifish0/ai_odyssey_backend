#!/usr/bin/env python3
"""
並發測試腳本：模擬15個用戶同時使用圖片生成服務
測試備用 API Key 切換機制和系統效能
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import List, Dict, Any
import statistics


class ConcurrentImageGenerator:
    def __init__(self, base_url: str, auth_token: str, num_users: int = 15):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.num_users = num_users
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        
    async def generate_single_image(self, session: aiohttp.ClientSession, user_id: int, prompt: str) -> Dict[str, Any]:
        """單個用戶生成圖片"""
        url = f"{self.base_url}/module1/generate-recipe-image"
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.auth_token}',
            'Content-Type': 'application/json'
        }
        data = {
            'prompt': prompt
        }
        
        start_time = time.time()
        try:
            async with session.post(url, headers=headers, json=data) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    return {
                        'user_id': user_id,
                        'status': 'success',
                        'response_time': response_time,
                        'status_code': response.status,
                        'image_url': result_data.get('image_url'),
                        'prompt': result_data.get('prompt'),
                        'model_used': result_data.get('model_used'),
                        'generation_time': result_data.get('generation_time'),
                        'error': None
                    }
                else:
                    error_text = await response.text()
                    return {
                        'user_id': user_id,
                        'status': 'error',
                        'response_time': response_time,
                        'status_code': response.status,
                        'image_url': None,
                        'prompt': prompt,
                        'model_used': None,
                        'generation_time': None,
                        'error': error_text
                    }
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return {
                'user_id': user_id,
                'status': 'error',
                'response_time': response_time,
                'status_code': None,
                'image_url': None,
                'prompt': prompt,
                'model_used': None,
                'generation_time': None,
                'error': str(e)
            }
    
    async def run_concurrent_test(self, prompts: List[str] = None):
        """執行並發測試"""
        if prompts is None:
            # 預設的測試提示詞
            prompts = [
                "宮保雞丁", "麻婆豆腐", "糖醋里脊", "魚香肉絲", "回鍋肉",
                "水煮魚", "東坡肉", "紅燒肉", "清蒸魚", "白切雞",
                "蒜蓉炒青菜", "番茄炒蛋", "青椒炒肉", "酸菜魚", "辣子雞"
            ]
        
        print(f"🚀 開始並發測試 - {self.num_users} 個用戶同時請求")
        print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 目標URL: {self.base_url}")
        print(f"📝 測試提示詞數量: {len(prompts)}")
        print("-" * 80)
        
        self.start_time = time.time()
        
        # 創建並發任務
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(self.num_users):
                prompt = prompts[i % len(prompts)]
                task = self.generate_single_image(session, i + 1, prompt)
                tasks.append(task)
            
            # 同時執行所有任務
            print("📤 發送並發請求...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 處理結果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.results.append({
                        'user_id': i + 1,
                        'status': 'exception',
                        'response_time': 0,
                        'status_code': None,
                        'image_url': None,
                        'prompt': prompts[i % len(prompts)],
                        'model_used': None,
                        'generation_time': None,
                        'error': str(result)
                    })
                else:
                    self.results.append(result)
        
        self.end_time = time.time()
        self.print_results()
    
    def print_results(self):
        """打印測試結果"""
        total_time = self.end_time - self.start_time
        successful_requests = [r for r in self.results if r['status'] == 'success']
        failed_requests = [r for r in self.results if r['status'] != 'success']
        
        print("\n" + "=" * 80)
        print("📊 測試結果摘要")
        print("=" * 80)
        print(f"⏱️  總執行時間: {total_time:.2f} 秒")
        print(f"👥 總請求數: {len(self.results)}")
        print(f"✅ 成功請求: {len(successful_requests)}")
        print(f"❌ 失敗請求: {len(failed_requests)}")
        print(f"📈 成功率: {(len(successful_requests) / len(self.results) * 100):.1f}%")
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            print(f"\n⏱️  響應時間統計:")
            print(f"   平均響應時間: {statistics.mean(response_times):.2f} 秒")
            print(f"   最快響應時間: {min(response_times):.2f} 秒")
            print(f"   最慢響應時間: {max(response_times):.2f} 秒")
            print(f"   中位數響應時間: {statistics.median(response_times):.2f} 秒")
        
        print(f"\n📋 詳細結果:")
        print("-" * 80)
        for result in self.results:
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} 用戶 {result['user_id']:2d}: {result['prompt']:10s} | "
                  f"狀態: {result['status']:8s} | "
                  f"響應時間: {result['response_time']:6.2f}s | "
                  f"狀態碼: {result['status_code'] or 'N/A'}")
            
            if result['error']:
                print(f"   錯誤: {result['error']}")
        
        # 分析可能的 API Key 切換
        if successful_requests:
            print(f"\n🔑 API Key 使用分析:")
            print("-" * 80)
            # 檢查是否有響應時間明顯不同的請求，可能表示使用了備用 API Key
            response_times = [r['response_time'] for r in successful_requests]
            avg_time = statistics.mean(response_times)
            std_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
            
            slow_requests = [r for r in successful_requests if r['response_time'] > avg_time + std_time]
            fast_requests = [r for r in successful_requests if r['response_time'] < avg_time - std_time]
            
            if slow_requests:
                print(f"🐌 較慢的請求 (可能使用了備用 API Key):")
                for req in slow_requests:
                    print(f"   用戶 {req['user_id']}: {req['prompt']} - {req['response_time']:.2f}s")
            
            if fast_requests:
                print(f"⚡ 較快的請求 (可能使用了主要 API Key):")
                for req in fast_requests:
                    print(f"   用戶 {req['user_id']}: {req['prompt']} - {req['response_time']:.2f}s")


async def main():
    """主函數"""
    # 配置參數
    BASE_URL = "http://127.0.0.1:8000"
    AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ3aWxsIiwiZXhwIjoxNzU2MzU0MzMzfQ.Ut8b_60rVsSqyh06_3Oemw8sVTQxvJf3IW1DwUgjqow"
    NUM_USERS = 60
    
    # 創建測試器
    tester = ConcurrentImageGenerator(BASE_URL, AUTH_TOKEN, NUM_USERS)
    
    # 執行測試
    await tester.run_concurrent_test()


if __name__ == "__main__":
    # 檢查依賴
    try:
        import aiohttp
        import statistics
    except ImportError as e:
        print(f"❌ 缺少依賴套件: {e}")
        print("請安裝: pip install aiohttp")
        exit(1)
    
    print("🔧 並發圖片生成測試工具")
    print("=" * 50)
    
    # 執行測試
    asyncio.run(main())
