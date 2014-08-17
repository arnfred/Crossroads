import web
import urllib

from recommender import query, search
from recommender.recommender import ArXivRecommender

# Define pages
urls = (
  '/d/(.[a-z0-9\./]*)-([0-9]*)/?', 'document',
  '/?', 'index',
  '/search/(.*)/?', 'search_query',
  '/(.[a-z0-9\.]*)/?', 'index_id'
)
render = web.template.render('templates/')

# Init recommender
recommender = ArXivRecommender('recommender/data/recommender.h5', 'recommender/data/arxiv.db')
# Init search engine
search_engine = search.ArXivSearchEngine(recommender)

# Index page displays start page
class index :
	def GET(self):
		return render.main("undefined")

class index_id :
	def GET(self, paper_id) :
		return render.main(str(paper_id))

# Returns the nearest neighbors of a given id
class document :
	def GET(self, doc_id, k) :
		return query.center(recommender, doc_id, int(k))

class search_query :
	def GET(self, decoded_terms) :
		search_input = urllib.unquote(decoded_terms)
		return search_engine.query(search_input)


# Run the app
if __name__ == "__main__" :
	# Init app
	app = web.application(urls, globals())
	app.run()

