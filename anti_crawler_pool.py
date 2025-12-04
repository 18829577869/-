"""
反爬虫工具模块
实现Cookie池、UA池、代理池等功能，用于绕过反爬限制
"""

import random
import time
import requests
from typing import List, Optional, Dict
import warnings


class AntiCrawlerPool:
    """反爬虫池管理器"""
    
    def __init__(self):
        """初始化反爬虫池"""
        self.user_agents = self._init_user_agents()
        self.cookies_pool = []  # Cookie池（可以动态添加）
        self.proxies_pool = []  # 代理池（可以动态添加）
        self.current_proxy_index = 0
        self.current_ua_index = 0
        self.current_cookie_index = 0
        
    def _init_user_agents(self) -> List[str]:
        """初始化User-Agent池"""
        return [
            # Chrome Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            # Chrome Mac
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            # Firefox Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            # Firefox Mac
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
            # Edge
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            # Safari
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            # 移动端
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Linux; Android 13; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
        ]
    
    def get_random_user_agent(self) -> str:
        """获取随机User-Agent"""
        return random.choice(self.user_agents)
    
    def get_rotating_user_agent(self) -> str:
        """获取轮换的User-Agent"""
        ua = self.user_agents[self.current_ua_index]
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
        return ua
    
    def add_proxy(self, proxy: str):
        """
        添加代理到代理池
        
        参数:
            proxy: 代理地址，格式如 'http://user:pass@host:port' 或 'socks5://host:port'
        """
        if proxy not in self.proxies_pool:
            self.proxies_pool.append(proxy)
    
    def add_proxies(self, proxies: List[str]):
        """批量添加代理"""
        for proxy in proxies:
            self.add_proxy(proxy)
    
    def get_rotating_proxy(self) -> Optional[Dict[str, str]]:
        """
        获取轮换的代理
        
        返回:
            代理字典，格式如 {'http': 'http://proxy:port', 'https': 'https://proxy:port'}
            如果没有代理，返回None
        """
        if not self.proxies_pool:
            return None
        
        proxy_url = self.proxies_pool[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies_pool)
        
        # 转换为requests格式
        if proxy_url.startswith('http://') or proxy_url.startswith('https://'):
            return {
                'http': proxy_url,
                'https': proxy_url
            }
        elif proxy_url.startswith('socks5://'):
            # SOCKS5代理需要安装requests[socks]
            return {
                'http': proxy_url,
                'https': proxy_url
            }
        else:
            # 默认使用http
            return {
                'http': f'http://{proxy_url}',
                'https': f'https://{proxy_url}'
            }
    
    def get_random_proxy(self) -> Optional[Dict[str, str]]:
        """获取随机代理"""
        if not self.proxies_pool:
            return None
        
        proxy_url = random.choice(self.proxies_pool)
        if proxy_url.startswith('http://') or proxy_url.startswith('https://'):
            return {
                'http': proxy_url,
                'https': proxy_url
            }
        elif proxy_url.startswith('socks5://'):
            return {
                'http': proxy_url,
                'https': proxy_url
            }
        else:
            return {
                'http': f'http://{proxy_url}',
                'https': f'https://{proxy_url}'
            }
    
    def add_cookie(self, cookie: str):
        """添加Cookie到Cookie池"""
        if cookie not in self.cookies_pool:
            self.cookies_pool.append(cookie)
    
    def add_cookies(self, cookies: List[str]):
        """批量添加Cookie"""
        for cookie in cookies:
            self.add_cookie(cookie)
    
    def get_rotating_cookie(self) -> Optional[str]:
        """获取轮换的Cookie"""
        if not self.cookies_pool:
            return None
        
        cookie = self.cookies_pool[self.current_cookie_index]
        self.current_cookie_index = (self.current_cookie_index + 1) % len(self.cookies_pool)
        return cookie
    
    def get_random_cookie(self) -> Optional[str]:
        """获取随机Cookie"""
        if not self.cookies_pool:
            return None
        return random.choice(self.cookies_pool)
    
    def random_delay(self, min_seconds: float = 0.5, max_seconds: float = 2.0):
        """随机延迟，模拟人类行为"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
    
    def get_headers(self, use_rotating: bool = True) -> Dict[str, str]:
        """
        获取请求头（包含随机UA）
        
        参数:
            use_rotating: 是否使用轮换模式（True）还是随机模式（False）
        
        返回:
            请求头字典
        """
        if use_rotating:
            ua = self.get_rotating_user_agent()
        else:
            ua = self.get_random_user_agent()
        
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        
        # 添加Cookie（如果有）
        cookie = self.get_rotating_cookie() if use_rotating else self.get_random_cookie()
        if cookie:
            headers['Cookie'] = cookie
        
        return headers
    
    def setup_requests_session(self, use_proxy: bool = True, use_rotating: bool = True) -> requests.Session:
        """
        创建配置好的requests Session
        
        参数:
            use_proxy: 是否使用代理
            use_rotating: 是否使用轮换模式
        
        返回:
            配置好的Session对象
        """
        session = requests.Session()
        
        # 设置请求头
        session.headers.update(self.get_headers(use_rotating))
        
        # 设置代理
        if use_proxy:
            proxy = self.get_rotating_proxy() if use_rotating else self.get_random_proxy()
            if proxy:
                session.proxies.update(proxy)
        
        # 设置超时
        session.timeout = 10
        
        return session


# 全局实例
_global_pool = None

def get_global_pool() -> AntiCrawlerPool:
    """获取全局反爬虫池实例"""
    global _global_pool
    if _global_pool is None:
        _global_pool = AntiCrawlerPool()
    return _global_pool


def setup_akshare_environment(pool: Optional[AntiCrawlerPool] = None):
    """
    设置AkShare的环境变量（用于设置代理和UA）
    
    参数:
        pool: 反爬虫池实例，如果为None则使用全局实例
    """
    import os
    
    if pool is None:
        pool = get_global_pool()
    
    # 设置代理环境变量（如果代理池不为空）
    proxy = pool.get_rotating_proxy()
    if proxy:
        # 提取代理地址（去掉协议前缀）
        proxy_url = proxy.get('http', '') or proxy.get('https', '')
        if proxy_url:
            # 设置环境变量（某些库会读取这些变量）
            if proxy_url.startswith('http://'):
                os.environ['HTTP_PROXY'] = proxy_url
                os.environ['HTTPS_PROXY'] = proxy_url.replace('http://', 'https://')
            elif proxy_url.startswith('https://'):
                os.environ['HTTPS_PROXY'] = proxy_url
                os.environ['HTTP_PROXY'] = proxy_url.replace('https://', 'http://')
            elif proxy_url.startswith('socks5://'):
                os.environ['HTTP_PROXY'] = proxy_url
                os.environ['HTTPS_PROXY'] = proxy_url
    
    # 设置User-Agent环境变量
    ua = pool.get_rotating_user_agent()
    os.environ['USER_AGENT'] = ua


def monkey_patch_requests(pool: Optional[AntiCrawlerPool] = None):
    """
    对requests库进行monkey patch，自动添加UA和代理
    
    参数:
        pool: 反爬虫池实例，如果为None则使用全局实例
    """
    if pool is None:
        pool = get_global_pool()
    
    original_get = requests.get
    original_post = requests.post
    
    def patched_get(url, **kwargs):
        """带反爬虫功能的get请求"""
        # 添加UA
        if 'headers' not in kwargs:
            kwargs['headers'] = {}
        kwargs['headers'].update(pool.get_headers())
        
        # 添加代理
        if 'proxies' not in kwargs:
            proxy = pool.get_rotating_proxy()
            if proxy:
                kwargs['proxies'] = proxy
        
        # 添加随机延迟
        pool.random_delay(0.3, 1.0)
        
        return original_get(url, **kwargs)
    
    def patched_post(url, **kwargs):
        """带反爬虫功能的post请求"""
        # 添加UA
        if 'headers' not in kwargs:
            kwargs['headers'] = {}
        kwargs['headers'].update(pool.get_headers())
        
        # 添加代理
        if 'proxies' not in kwargs:
            proxy = pool.get_rotating_proxy()
            if proxy:
                kwargs['proxies'] = proxy
        
        # 添加随机延迟
        pool.random_delay(0.3, 1.0)
        
        return original_post(url, **kwargs)
    
    # 应用patch
    requests.get = patched_get
    requests.post = patched_post

