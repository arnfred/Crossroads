"""
Database management utility inspired by the work of kijun for the 
Caltech centrality project at https://github.com/centrality/arxiv
"""

import re
import time
from datetime import datetime
from time import mktime
import sqlalchemy
import sqlalchemy.types
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker


# Reg exp for parsing
STRCHAR = re.compile("[^a-zA-Z\s]")
FIRST_INT = re.compile(r'[^a-zA-Z\s\.]')
STRCHAR2 = re.compile("[^a-zA-Z]")
IDREGEX = re.compile(r'v[0-9]+$')

# Declare stuff for sqlachemy ORM
# engine = sqlalchemy.create_engine('sqlite:////Volumes/MyPassport/data/arxiv.db')
engine = sqlalchemy.create_engine('sqlite:///arxiv.db')
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


class Article(Base):
    """
    Base class for the sqlite 'Article' table representing a tuple for a 
    crawled arXiv paper
    """
    __tablename__ = 'Articles'
    id = Column(String(15), primary_key=True)   # Article ID
    title = Column(Text)                        # Title
    abstract = Column(Text)                     # Abstract
    journal = Column(String(255))               # Journal (if given)
    published_at = Column(DateTime)             # Publication date
    updated_at = Column(DateTime)               # Last update date
    authors = Column(Text)                      # List of authors
    prim_category = Column(String(50))          # Primary category
    categories = Column(Text)                   # List of categories
    abs_link = Column(String(255))              # Link to arXiv page
    pdf_link = Column(String(255))              # Link to pdf page
    
    def from_atom_entry(self, entry):
        """
        Fill an Article object from a parsed entry of feedparser
        """
        self.id = IDREGEX.sub('', entry.id[21:])
        self.title = entry.title
        self.abstract = entry.summary
        if 'arxiv_journal_ref' in entry:
            self.journal = STRCHAR2.sub('', entry['arxiv_journal_ref'])
        self.published_at = datetime.fromtimestamp(mktime(entry['published_parsed']))
        self.updated_at = datetime.fromtimestamp(mktime(entry['updated_parsed']))
        names = [a['name'] for a in entry['authors']]
        self.authors = '|'.join(names)
        self.prim_category = entry['arxiv_primary_category']['term']
        self.categories = '|'.join([t['term'] for t in entry['tags']])
        for link in entry.links:
            if link.rel == 'alternate':
                self.abs_link = link.href
            elif link.title == 'pdf':
               self.pdf_link = link.href

    def __repr__(self):
        s = '''<Article(id = %s, 
        title = %s, 
        abstract = %s,
        published_at = %s, 
        updated_at = %s, 
        authors = %s, 
        prim_category = %s, 
        categories = %s
        abs_link = %s,
        pdf_link = %)s>''' % (self.id,
                            self.title[:30]+'...',
                            self.abstract[:30]+'...',
                            str(self.published_at),
                            str(self.updated_at),
                            self.authors,
                            self.prim_category,
                            self.categories,
                            self.abs_link,
                            self.pdf_link)
        return s.encode('utf8')


def create_tables():
    Base.metadata.create_all(engine)
