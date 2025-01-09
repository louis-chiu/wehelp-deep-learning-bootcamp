from urllib.request import urlopen
from urllib.parse import urlencode
from collections.abc import Generator
import json


class Config:
    BASE_URL = "https://ecshweb.pchome.com.tw/search/v4.3/all/results"
    ASUS_CATEGORY_ID = "DSAA31"
    PAGE_SIZE = 30
    I5_PROCESSOR_ATTRIBUTE_ID = "G26I2272"


class MathUtils:
    @staticmethod
    def mean(numbers: list[int | float]) -> int | float:
        if len(numbers) == 0:
            raise ValueError("List of numbers may not be empty.")

        return sum(numbers) / len(numbers)

    @staticmethod
    def stdev(numbers: list[int | float]) -> int | float:
        if len(numbers) == 0:
            raise ValueError("List of number may not be empty.")

        mu = MathUtils.mean(numbers)
        return (sum([(number - mu) ** 2 for number in numbers]) / len(numbers)) ** 0.5

    @staticmethod
    def zscore(number: int | float, mu: int | float, sigma: int | float) -> int | float:
        if sigma == 0:
            raise ValueError("Sigma must not be 0.")

        return (number - mu) / sigma


class PersonalComputerFetcher:
    @staticmethod
    def get_url(base_url, **query_params):
        encoded_params = urlencode(query_params)
        return f"{base_url}?{encoded_params}"

    @staticmethod
    def fetch_page(url):
        with urlopen(url) as response:
            body = response.read()
            return json.loads(body)

    @staticmethod
    def fetch_total_pages(**query_params):
        url = PersonalComputerFetcher.get_url(
            Config.BASE_URL, page_number=0, **query_params
        )
        first_page = PersonalComputerFetcher.fetch_page(url)
        return first_page.get("TotalPage"), Utils.safe_get(first_page, "Prods", [])

    @staticmethod
    def get_pc_pages(**query_params) -> Generator[list, None, None]:
        total_page, first_page = PersonalComputerFetcher.fetch_total_pages(
            **query_params
        )
        yield first_page

        if total_page <= 1:
            return

        for page_number in range(1, total_page):
            url = PersonalComputerFetcher.get_url(
                Config.BASE_URL, page=page_number, **query_params
            )
            next_page = PersonalComputerFetcher.fetch_page(url)
            yield Utils.safe_get(next_page, "Prods", [])


class FileUtils:
    @staticmethod
    def write_to_file(file_path, lines):
        with open(file_path, "w+") as file:
            file.write("\n".join(lines))


class Utils:
    @staticmethod
    def safe_get(dict_, key, default_value):
        return dict_.get(key, default_value) or default_value


def main():
    # Task 1
    products = [
        product
        for products_ in PersonalComputerFetcher.get_pc_pages(
            cateid=Config.ASUS_CATEGORY_ID, pageCount=Config.PAGE_SIZE
        )
        for product in products_
    ]
    FileUtils.write_to_file(
        "./products.txt", [product.get("Id") for product in products]
    )

    # Task 2
    best_product_ids = [
        product.get("Id")
        for product in products
        if Utils.safe_get(product, "ratingValue", 0) > 4.9
        and Utils.safe_get(product, "reviewCount", 0) >= 1
    ]
    FileUtils.write_to_file("./best-products.txt", best_product_ids)

    # Task 3
    pcs_with_i5_processor = [
        Utils.safe_get(product, "Price", 0)
        for products_ in PersonalComputerFetcher.get_pc_pages(
            cateid=Config.ASUS_CATEGORY_ID,
            pageCount=Config.PAGE_SIZE,
            attr=Config.I5_PROCESSOR_ATTRIBUTE_ID,
        )
        for product in products_
    ]
    print(f"{MathUtils.mean(pcs_with_i5_processor):,.2f}")

    # Task 4
    price_of_products = [Utils.safe_get(product, "Price", 0) for product in products]
    average_price = MathUtils.mean(price_of_products)
    stdev_of_price = MathUtils.stdev(price_of_products)
    FileUtils.write_to_file(
        "./standardization.csv",
        [
            f"{product.get('Id')},{Utils.safe_get(product, 'Price', 0)},{MathUtils.zscore(Utils.safe_get(product, 'Price', 0), average_price, stdev_of_price):.2}"
            for product in products
        ],
    )


if __name__ == "__main__":
    main()

# https://stackoverflow.com/questions/17777845/python-requests-arguments-dealing-with-api-pagination
