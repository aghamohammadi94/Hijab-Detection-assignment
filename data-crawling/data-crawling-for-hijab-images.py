
# crawl data from Google for hijab images
# import library
from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    parser_threads=2,
    downloader_threads=2,
    storage={'root_dir': 'downloads'}
)

keywords = ['hijab'] # your keywords here
num_images = 1000 # number of images per keyword

for keyword in keywords:
    google_crawler.crawl(keyword=keyword, max_num=num_images)
    
