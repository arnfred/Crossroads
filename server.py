import web
import query

# Define pages
urls = (
  '/', 'index',
  '/d/(.*)', 'document'
)
render = web.template.render('templates/')


# Index page displays start page
class index:
    def GET(self):
        return render.main()

# Returns the nearest neighbors of a given id
class document:
    def GET(self, doc_id):
        return query.center(doc_id, 10)

# Run the app
if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()


