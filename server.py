import web
import urllib
import query
from search import search_papers

# Define pages
urls = (
  '/d/(.[a-z0-9\.]*)/([0-9]*)/', 'document',
  '/', 'index',
  '/search/(.*)', 'search',
  '/(.[a-z0-9\.]*)/', 'index_id'
)
render = web.template.render('templates/')


# Index page displays start page
class index :
    def GET(self):
        return render.main("1401.6060")

class index_id :
    def GET(self, paper_id) :
        return render.main(str(paper_id))

# Returns the nearest neighbors of a given id
class document :
    def GET(self, doc_id, k) :
        return query.center(doc_id, int(k))

class search :
    def GET(self, decoded_terms) :
        search_input = urllib.unquote(decoded_terms).decode('utf8')
        return search_papers(search_input)


# Run the app
if __name__ == "__main__" :
    app = web.application(urls, globals())
    app.run()


