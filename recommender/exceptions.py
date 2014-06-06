

class UnknownIDException(Exception):
	"""
	Exception raised if a unknown paper id is queried
	"""
	
	def __init__(self, paper_id):
		self.paper_id = paper_id

	def __str__(self):
		return repr("UnknownIDException: Unknown paper id %s" % self.paper_id)


class UnknownAuthorException(Exception):
	"""
	Exception raised if a unknown paper id is queried
	"""
	
	def __init__(self, author):
		self.author = author

	def __str__(self):
		return repr("UnknownAuthorException: Unknown author %s" % self.author)
