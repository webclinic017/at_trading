from newscatcher import Newscatcher, urls

nc = Newscatcher(website='wsj.com')
results = nc.get_news()
#
# # results.keys()
# # 'url', 'topic', 'language', 'country', 'articles'
#
# # Get the articles
# articles = results['articles']
#
# first_article_summary = articles[0]['summary']
# first_article_title = articles[0]['title']
#
finance = urls(topic='finance')
#
# # URLs by COUNTRY
# american_urls = urls(country='US')
#
# # URLs by LANGUAGE
# english_urls = urls(language='en')
