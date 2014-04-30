import web
import query

# Define pages
urls = (
  '/d/(.[a-z0-9\.]*)/([0-9]*)/', 'document',
  '/', 'index',
  '/(.[a-z0-9\.]*)/', 'index_id'
)
render = web.template.render('templates/')


# Index page displays start page
class index:
    def GET(self):
        return render.main("1304.6026")

class index_id:
    def GET(self, paper_id):
        return render.main(str(paper_id))

# Returns the nearest neighbors of a given id
class document:
    def GET(self, doc_id, k):
        print(k)
        return query.center(doc_id, int(k))

# Run the app
if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()


