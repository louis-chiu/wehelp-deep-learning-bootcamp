import scrapy
from scrapy.crawler import CrawlerProcess
from multiprocessing import Process
from pathlib import Path
import random
from scrapy.utils.log import configure_logging
import logging
import csv
from collections import deque

OUTPUT_DIR = Path("./data")
LOG_DIR = Path("./logs")
BASE_URL = "https://www.ptt.cc{}"
BOARD_NAMES = [
    "baseball",
    "Boy-Girl",
    "c_chat",
    "hatepolitics",
    "Lifeismoney",
    "Military",
    "pc_shopping",
    "stock",
    "Tech_Job",
]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
]
COOOKIES = {"over18": "1"}


class MultiSpider(scrapy.Spider):
    name = "multi_spider"

    custom_settings = {
        "CONCURRENT_REQUESTS": 3,
        "DOWNLOAD_TIMEOUT": 5,
        "DOWNLOAD_DELAY": 2,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 520, 522, 524, 408, 429, 403],
    }

    def __init__(self, url, board_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [url]
        self.board_name = board_name
        self.proxy = None
        self.next_url = None

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                cookies=COOOKIES,
                headers={"User-Agent": random.choice(USER_AGENTS)},
                callback=self.parse,
                errback=self.handle_error,
            )

    def parse(self, response):
        for article in self._extract_articles(response):
            yield article

        self.next_url = self._get_next_page_url(response)
        if self.next_url:
            self.log(f"Next page url {self.next_url}")
            meta = {"proxy": self.proxy} if self.proxy else {}
            yield response.follow(
                self.next_url,
                headers={"User-Agent": random.choice(USER_AGENTS)},
                meta=meta,
                callback=self.parse,
                errback=self.handle_error,
            )
        else:
            self.log(f"{self.board_name}: Reached last page. Crawler job finished.")
            self.write_starting_url(response.url)

    def _extract_articles(self, response):
        for row_entry in response.xpath(
            '//*[@id="main-container"]/div[2]/div[@class="r-ent"]'
        ):
            metadata = row_entry.xpath('./div[@class="meta"]')
            self.log(
                f"{self.board_name}: Page {self.crawler.stats.get_value('downloader/request_count', 0)} Items Count {self.crawler.stats.get_value('item_scraped_count', 0)}"
            )

            article_link = row_entry.xpath('./div[@class="title"]/a')
            if not article_link:
                continue

            yield {
                "link": article_link.attrib["href"],
                "title": article_link.xpath("./text()").get(),
                "author": metadata.xpath('./div[@class="author"]/text()').get(),
                "num_recommendations": row_entry.xpath(
                    './div[@class="nrec"]/span/text()'
                ).get(),
                "date": metadata.xpath('./div[@class="date"]/text()').get(),
            }

    def _get_next_page_url(self, response):
        return response.xpath('//*[@id="action-bar-container"]/div/div[2]/a[2]').attrib[
            "href"
        ]

    def close(self, reason):
        print(f"{self.board_name} Crawler: {self.start_urls[0]} Finished. ")

        if reason == "shutdown" or "signal" in reason:
            self.write_starting_url(BASE_URL.format(self.next_url))

    def handle_error(self, failure):
        request = failure.request
        self.logger.error(f"Request failed: <{request.url}> {failure.value}")
        self.write_starting_url(request.url)

    def write_starting_url(self, url):
        with open(OUTPUT_DIR / f".{self.board_name}", "w", encoding="utf-8") as file:
            file.write(f"{url}")


def run_spider(url, board_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    output_path = str(OUTPUT_DIR / f"{board_name}.csv")
    log_file = str(LOG_DIR / f"{board_name}.log")

    configure_logging(install_root_handler=False)
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )

    process = CrawlerProcess(
        {
            "FEEDS": {
                output_path: {
                    "format": "csv",
                    "overwrite": False,
                    "append": True,
                    "item_export_kwargs": {
                        "include_headers_line": not Path(output_path).exists()
                    },
                },
            },
            "LOG_STDOUT": True,
        }
    )
    process.crawl(MultiSpider, url=url, output_file=output_path, board_name=board_name)
    process.start()


def is_last_page(url: str) -> bool:
    return "index1.html" in url


def main():
    processes = []
    for board_name in BOARD_NAMES:
        if (starting_url_file := OUTPUT_DIR / f".{board_name}").exists():
            with open(starting_url_file) as file2:
                starting_url = file2.readline()
                if is_last_page(starting_url):
                    continue
                process = Process(
                    target=run_spider,
                    args=(starting_url, board_name),
                )
        else:
            process = Process(
                target=run_spider,
                args=(BASE_URL.format(f"/bbs/{board_name}"), board_name),
            )

        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print("Crawler jobs finished!")


if __name__ == "__main__":
    main()
