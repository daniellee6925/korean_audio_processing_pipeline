import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Optional, List
from bs4 import BeautifulSoup
from tqdm import async_tqdm
import re
import json
import os
from tqdm.asyncio import tqdm as async_tqdm
from rate_limiter import RateLimiter
from functools import wraps
import time
from loguru import logger
from seleniumwire import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} runtime: {end - start:.4f} seconds")
        return result

    return wrapper


class DriverScraper:
    def __init__(
        self,
        urls: list[str],
        text_div: str,
        audio_extensions: list[str] = ["mp3", "wav", "flac"],
        save_extension: str = "wav",
        save_dir="audio_files_3",
        # Rate limiting configs
        max_concurrent: int = 10,
        requests_per_second: float = 5.0,
        burst_size: int = 20,
        # Retry configs
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        # Timeout configs
        connect_timeout: int = 10,
        read_timeout: int = 30,
        chrome_driver_path: str = "/opt/homebrew/bin/chromedriver",
        headless: bool = True,
        dev: bool = False,
        button_timeout: int = 10,
        page_timeout: int = 10,
    ):
        self.urls = urls
        self.text_div = text_div
        self.audio_extensions = audio_extensions
        self.save_dir = save_dir
        self.save_extension = save_extension

        # Rate limiting
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)  # limits num of concurrent operations
        self.rate_limiter = RateLimiter(requests_per_second)
        self.burst_size = burst_size

        # Retry settings
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

        # Timeouts
        self.timeout = aiohttp.ClientTimeout(
            total=connect_timeout + read_timeout,
            connect=connect_timeout,
            sock_read=read_timeout,
        )
        self.button_timeout = button_timeout
        self.page_timeout = page_timeout

        self.session: Optional[aiohttp.ClientSession] = None  # hold http session to track stats

        # Statistics
        self.stats = {"success": 0, "failed": 0, "retried": 0, "rate_limited": 0}

        # chrome driver
        self.chrome_driver_path = chrome_driver_path
        self.headless = headless
        self.dev = dev
        self.driver: Optional[webdriver.Chrome] = None

    def initialize_driver(
        self,
    ) -> None:
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        if self.dev:
            options.add_experimental_option("detach", True)  # use for dev
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        service = Service(self.chrome_driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        self.driver = driver

    def get_buttons(self, button_flag, timeout=10):
        """Wait until all play buttons are present in the DOM."""
        try:
            buttons = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_all_elements_located(button_flag)
            )

            return buttons
        except Exception as e:
            logger.error(f"Error Finding Buttons: {e}")
        return []

    def click_button_and_capture_url(self, driver, button):
        """Click a button and capture audio URLs triggered by the click."""
        self.driver.requests.clear()

        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
            driver.execute_script("arguments[0].click();", button)

            audio_urls = set()
            for request in driver.requests:
                if request.response and "audio" in request.response.headers.get("Content-Type", ""):
                    url = request.url
                    if url not in audio_urls:
                        audio_urls.add(url)
            return list(audio_urls)
        except Exception as e:
            logger.error(f"Error clicking button: {e}")
            return []

    async def download_single_audio(self, session: aiohttp.ClientSession, url: str, filepath: Path):
        """Download a single audio file with retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    response.raise_for_status()
                    content = await response.read()

                    async with aiofiles.open(filepath, "wb") as f:
                        await f.write(content)

                    logger.info(f"Downloaded {filepath}")
                    self.stats["success"] += 1
                    return True
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Download failed (attempt {attempt + 1}/{self.max_retries}): {url}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    self.stats["retried"] += 1
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to download {url} after {self.max_retries} attempts: {e}")
                    self.stats["failed"] += 1
                    return False

    async def download_audio(self, audio_urls: List[str], save_path: Path) -> None:
        """Download audio files concurrently in batches"""
        batch_size = min(self.burst_size, len(audio_urls))
        for i in range(0, len(audio_urls), batch_size):
            batch = audio_urls[i : i + batch_size]
            tasks = []

            for i, url in enumerate(batch):
                filename = save_path / f"audio_{i}.{self.save_extension}"
                tasks.append(self.download_audio(url, filename))

            await asyncio.gather(*tasks, return_exceptions=True)

    # ////////////////////////////////////////////////

    async def _make_request_with_retry(
        self, url: str, request_type: str = "get", **kwargs
    ) -> Optional[aiohttp.ClientResponse]:
        """Make HTTP request with exponential backoff retry"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Rate limiting before request
                await self.rate_limiter.acquire()

                async with self.semaphore:
                    if request_type == "get":
                        response = await self.session.get(url, **kwargs)
                    else:
                        raise ValueError(f"Unknown request type: {request_type}")

                    # Handle rate limiting responses
                    if response.status == 429:  # tooy many requests
                        self.stats["rate_limited"] += 1
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = float(retry_after)
                        else:
                            wait_time = self.retry_delay * (self.backoff_factor**attempt)

                        logger.debug(f"Rate limited on {url}, waiting {wait_time:.3f}s")
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    return response

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (self.backoff_factor**attempt)
                    logger.debug(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}): {url}"
                    )
                    logger.error(f"  Error: {e}. Retrying in {wait_time:.1f}s...")
                    self.stats["retried"] += 1
                    await asyncio.sleep(wait_time)
                else:
                    logger.debug(f"Max retries reached for {url}: {e}")
                    self.stats["failed"] += 1

        return None

    async def get_page_html(self, url: str) -> str:
        """Async fetch HTML content"""
        try:
            async with await self._make_request_with_retry(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/118.0.0.0 Safari/537.36"
                },
                timeout=self.timeout,
            ) as response:
                if response:
                    self.stats["success"] += 1
                    return await response.text()
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")

        return ""

    async def parse_html(self, url: str) -> Optional[BeautifulSoup]:
        """Parse HTML and return BeautifulSoup object"""
        html_source = await self.get_page_html(url)
        if not html_source:
            return None

        try:
            loop = asyncio.get_event_loop()
            soup = await loop.run_in_executor(None, BeautifulSoup, html_source, "html.parser")
            return soup
        except Exception as e:
            logger.error(f"Failed to parse HTML from {url}: {e}")
            return None

    def get_audio_urls(self, soup: BeautifulSoup) -> List[str]:
        """Extract audio URLs from soup"""
        audio_urls = set()
        candidate_tags = soup.find_all(["audio", "source", "a"])

        for tag in candidate_tags:
            for attr in ("src", "href"):
                src = tag.get(attr)
                if src and any(src.lower().endswith(f".{ext}") for ext in self.audio_extensions):
                    if src.startswith("//"):
                        src = "https:" + src
                    audio_urls.add(src)

        logger.info(f"Found {len(audio_urls)} audio links for this page.")
        return list(audio_urls)

    def get_meta_data(self, soup: BeautifulSoup) -> str:
        """Extract metadata from soup"""
        target_div = soup.find("div", class_=self.text_div)

        if target_div:
            paragraphs = [p.get_text(strip=True) for p in target_div.find_all("p")]
            korean_text = "\n\n".join(p for p in paragraphs if p)
            return korean_text.strip()
        return ""

    async def download_audio(self, url: str, file_path: Path) -> bool:
        """Download single audio file with retry logic"""
        try:
            response = await self._make_request_with_retry(url, timeout=self.timeout)
            if response:
                async with response:  # temp open network stream, guaranteed it gets closed
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await response.read())
                    logger.info(f"Downloaded {file_path}")
                    return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")

        return False

    async def save_audio(self, download_urls: List[str], save_path: Path) -> None:
        """Download all audio files concurrently with batching"""
        batch_size = min(self.burst_size, len(download_urls), 1)

        for i in range(0, len(download_urls), batch_size):
            batch = download_urls[i : i + batch_size]
            tasks = []

            for j, url in enumerate(batch):
                file_name = save_path / f"audio_{i+j+1}.{self.save_extension}"
                tasks.append(self.download_audio(url, file_name))

            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def extract_voice_idx(url: str) -> Optional[str]:
        """Extract voice index from URL"""
        match = re.search(r"/gig/(\d+)", url)
        if match:
            return match.group(1)
        return None

    async def save_meta_data(self, url: str, meta_data: str, save_path: Path) -> None:
        """Save metadata to JSON file"""
        voice_idx = self.extract_voice_idx(url)
        if not voice_idx:
            voice_idx = "no_idx"

        data = {
            "voice_idx": voice_idx,
            "url": url,
            "meta_data": meta_data,
        }

        json_file = save_path / f"{voice_idx}.json"
        async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=4))

    async def scrape_page(self, page_url: str, save_path: Path) -> None:
        """Scrape single page"""
        soup = await self.parse_html(page_url)
        if not soup:
            logger.debug(f"Skipping {page_url}, no HTML fetched.")
            return

        loop = asyncio.get_event_loop()
        download_urls = await loop.run_in_executor(None, self.get_audio_urls, soup)
        meta_data = await loop.run_in_executor(None, self.get_meta_data, soup)

        await asyncio.gather(
            self.save_audio(download_urls, save_path),
            self.save_meta_data(page_url, meta_data, save_path),
        )

    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300,
        )
        self.session = aiohttp.ClientSession(connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def process_all(self) -> None:
        """Process all URLs with progress bar"""
        tasks = []
        for url in self.urls:
            voice_idx = self.extract_voice_idx(url=url)
            save_path = Path(os.path.join(self.save_dir, voice_idx or "no_idx"))
            os.makedirs(save_path, exist_ok=True)
            tasks.append(self.scrape_page(url, save_path))

        # Process with progress bar
        for coro in async_tqdm.as_completed(tasks, desc="Extracting Audio Data"):
            await coro

        # Print statistics
        print(f"\n{'='*50}")
        print(f"Finished Processing {len(self.urls)} URLs")
        print(f"Success: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Retried: {self.stats['retried']}")
        print(f"Rate Limited: {self.stats['rate_limited']}")
        print(f"{'='*50}")

    @timer
    def run(self) -> None:
        """Entry point to run async scraper"""

        async def _run():
            async with self:  # calls __aenter__ and __aexit__
                await self.process_all()

        asyncio.run(_run())


# Usage examples with different configurations
if __name__ == "__main__":
    import csv

    csv_file = Path("kmong_urls.csv")
    urls = []

    with csv_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                urls.append(row[0])
    print(urls[:5])

    logger.info(f"Found {len(urls)} URLs")

    text_div = lambda x: x and "whitespace-pre-line" in x and "text-justify" in x

    balanced_scraper = AsyncAudioScraper(
        urls=urls,
        text_div=text_div,
        max_concurrent=10,
        requests_per_second=5.0,
        burst_size=20,
        max_retries=3,
    )

    balanced_scraper.run()
