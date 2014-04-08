"""
arXiv crawler using arXiv API. More information at http://arxiv.org/help/api/index
"""

import urllib
import calendar
import datetime
from dateutil import rrule
from dateutil.relativedelta import relativedelta
import time
import re
import sqlalchemy
from sqlalchemy.exc import IntegrityError

import feedparser

import database
reload(database)

BASE_URL = 'http://export.arxiv.org/api/query?' # Base api query url
WAIT_TIME = 3                                   # number of seconds to wait beetween calls
MAX_RESULTS = 1000  # API advises to fetch maximum 1000 articles at once    

# Opensearch metadata such as totalResults, startIndex, and itemsPerPage live in the opensearch 
# namespase. Some entry metadata lives in the arXiv namespace. This is a hack to expose both of 
# these namespaces in feedparser v4.1
feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'


class Crawler(object):

    def _execute_query(self, search_query, start=0, sortBy='lastUpdatedDate', sortOrder='ascending', minDate='000000000000', maxDate='999999999999'):
        """
        Execute a query on arXiv API

        search_query : str
            Search query
        start : int > 0
            Start index of the query (for paging, max_results is hardcoded) (default: 0)
        sortBy : str
            Argument by which the query is sorted (default: 'lastUpdatedDate')
        sortedOrder : str
            Order of the sort ('ascending' or 'descending') (default: 'ascending')
        minDate : str
            Formated start date of the query (format %Y%m%d%H%M) (default: '000000000000')
        maxDate : str
            Formated end date of the query (default: '999999999999')
        """
        # Build query
        query = 'search_query=(%s)+AND+lastUpdatedDate:[%s+TO+%s]&start=%d&max_results=%d&sortBy=%s&sortOrder=%s' % \
                                (search_query,minDate, maxDate, start,MAX_RESULTS, sortBy, sortOrder) 
        # Perform a GET request using the base_url and query
        response = urllib.urlopen(BASE_URL+query).read()
        return response

    def _time_slice_query(self, search_query, minDate, maxDate):
        """
        Retrieve entries for a given query in a time slice (since arXiv API 
        that does not return more than 50K articles per query)

        search_query : str
            Search query
        minDate : str
            Formated start date of the query (format %Y%m%d%H%M) (default: '000000000000')
        maxDate : str
            Formated end date of the query (default: '999999999999')
        """
        start = 0
        while True:
            start_time = time.time()
            n_fetched = 0
            n_errors = 0
            # Execute query and parse it
            response = self._execute_query(search_query, start=start, minDate=minDate, maxDate=maxDate)
            feed = feedparser.parse(response)
            # Iterate over article entries
            for entry in feed['entries']:
                try:
                    # Add them to the database
                    article = database.Article()
                    article.from_atom_entry(entry)
                    database.session.add(article)
                    database.session.commit()
                    n_fetched += 1
                except IntegrityError:
                    database.session.rollback()
                    n_errors += 1
                    pass
            start += len(feed['entries'])
            if start >= int(feed['feed']['opensearch_totalresults']):
                break
            # Waiting 3 sec between 2 queries to be nice with arXiv bandwith
            time.sleep(max(0, WAIT_TIME-time.time()+start_time))
        return n_fetched, n_errors      

    def query(self, search_query, datetimeRange=[datetime.datetime(1991,01,01),datetime.datetime.now()]):
        """
        Retrieve arXiv articles based on a search query for a given time range of lastUpdatedDate

        search_query : str
            Search query
        datetimeRange : list of datetime structures of length 2
            First element is the start date and second element is the end date
        """
        n_fetched_tot = 0
        n_errors_tot = 0

        database.create_tables() # Create the tables once and for all        
        startDate = datetimeRange[0]
        endDate = datetimeRange[1] - relativedelta(months=1)
        # Split query in monthly time slices
        for current_date in rrule.rrule(rrule.MONTHLY, dtstart=startDate, until=endDate):
                minDate = current_date
                maxDate = current_date + relativedelta(months=1)
                n_fetched, n_errors = self._time_slice_query(search_query, minDate.strftime("%Y%m%d%H%M"), maxDate.strftime("%Y%m%d%H%M"))
                n_fetched_tot += n_fetched
                n_errors_tot += n_errors
                print "Query[%s, %s]: %d articles fetched (%d in total), %d errors (%d in total)" % (minDate.strftime("%Y-%m-%d"), maxDate.strftime("%Y-%m-%d"), n_fetched, n_fetched_tot, n_errors, n_errors_tot)


if __name__ == "__main__":
    # Search query
    # search_query = 'cat:cs*+OR+cat:math*'
    search_query = 'cat:q-bio*+OR+cat:stat*'

    startDate = datetime.datetime(2000,1,1)
    endDate = datetime.datetime(2014,3,1)

    crawler = Crawler()
    crawler.query(search_query, [startDate, endDate])

