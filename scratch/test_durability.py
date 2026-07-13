import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promise_tracker_pipeline import build_search_fallback_url
from link_checker import check_url_live

def test_search_fallback():
    print("Running Search Fallback Tests...")
    
    # Test case 1: Headline exists and is clean
    u1 = "https://indianexpress.com/article/india/will-double-farmers-income-by-2022-says-pm-modi-4617098/"
    h1 = "Will double farmers' income by 2022, says PM Modi"
    fallback_1 = build_search_fallback_url(u1, h1)
    print(f"Fallback 1: {fallback_1}")
    assert "q=Will+double+farmers+income+by+2022+says+PM+Modi" in fallback_1
    
    # Test case 2: Headline is generic, derive from slug
    u2 = "https://www.thehindu.com/news/national/modi-government-promised-to-double-farm-income-by-2022-but-it-has-only-come-down-congress/article66318536.ece"
    h2 = "BJP Election Manifesto 2014"  # contains 'manifesto', generic keyword
    fallback_2 = build_search_fallback_url(u2, h2)
    print(f"Fallback 2: {fallback_2}")
    # Path: news/national/modi-government-promised-to-double-farm-income-by-2022-but-it-has-only-come-down-congress
    assert "q=news+national+modi+government+promised+to+double+farm+income+by+2022+but+it+has+only+come+down+congress" in fallback_2
    
    # Test case 3: Wayback URL with generic title
    u3 = "https://web.archive.org/web/20190208201614/http://www.bjp.org:80/manifesto2014"
    h3 = "BJP Election Manifesto 2014"
    fallback_3 = build_search_fallback_url(u3, h3)
    print(f"Fallback 3: {fallback_3}")
    assert "q=manifesto2014" in fallback_3

    print("All Search Fallback Tests Passed!")

def test_live_checker():
    print("Running Live Checker Tests...")
    # Check if wikipedia is live (wikipedia.org is pre-approved by sandbox proxy)
    is_live = check_url_live("https://en.wikipedia.org/")
    print(f"Wikipedia live: {is_live}")
    assert is_live == True
    print("Live Checker Tests Passed!")

if __name__ == "__main__":
    test_search_fallback()
