import numpy as np
from scipy.misc import logsumexp
from sys import stdout
import random, string
import time
import itertools







def select_topk_values(mydict, topk, descending=True):
	"""
	Select items with top values of a dictionary

	mydict		dictionary
	topk 		number of values to select
	descending	True to select the greatest values (by default), 
				False to select the smallest values
	"""

	d = {}
	count = 0
	for k, v in sorted( mydict.iteritems(), key = lambda(k,v):(v,k) , reverse=descending):
		if count < topk:
			d[k] = v
			count += 1
	return d


# ===================================================================================================================================
# ================= Random stuff generators utility functions
# ===================================================================================================================================


def random_string_generator(length=8):
	"""
	Return a random ascii string of digits and lower/upper case characters of a given length 
	"""
	chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
	return ''.join(random.choice(chars) for x in range(length))


# ===================================================================================================================================
# ================= Probability utility functions
# ===================================================================================================================================

class sampling(object):

	@staticmethod
	def GEM(m, pi, L):
		"""
		Return the samples from a stick-breaking construction GEM distribution [Pitman 2002] 
		truncated after L breaks. So that the L first components of the vector manifest the same
		behavior as the infinit stick-breaking construction and then normalized to 1.
		"""

		# Sample the V from a beta distribution
		v = np.random.beta(m*pi, (1-m)*pi, size=L)
		
		# Compute the GEM vector of probabilities
		theta = v
		theta[1:] *= np.cumprod(1-v[:L-1])

		theta /= np.sum(theta)
		return theta


	@staticmethod
	def sample_from_log_prob(A):
		"""
		Takes as input a log probability vector. The logarithm used is the natural logarithm.
		Returns a sampled position according to the corresponding log probabilities
		Uses the formula from
		http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
		"""
		# Compute log cdf
		C = [A[0]]
		for a in A[1:]:
			C.append( mymath.log_add(C[-1], a) )	
		# Normalize
		Cnorm = C - C[len(C)-1] 
		# Sample a uniform number in the interval (0,1) and take its log
		r = np.log( np.random.uniform() )
		# Get the corresponding position in the cdf
		pos = np.searchsorted(Cnorm,r)
		return pos


# ===================================================================================================================================
# ================= Math utility functions
# ===================================================================================================================================

class mymath(object):


	@staticmethod
	def cosine_similarity(a,b):
		"""
		Cosine similarity between two vectors
		"""
		return np.dot(a,b) / (np.linalg.norm(a,2)*np.linalg.norm(b,2))

	@staticmethod
	def KL(p,q):
		"""
		Compute the Kullback Lieber divergence D( p || q )
		Both distributions p and q must have the same shape
		(if they are matrices, KL is done by columns)
		"""
		p = np.array(p, dtype=np.float)
		q = np.array(q, dtype=np.float)
		q[q < 1e-50] = 1e-50
		p[p < 1e-50] = 1e-50
		return np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=0)

	@staticmethod
	def JSD(p,q):
		"""
		Compute the Jensen-Shannon divergence JS( p || q )
		This function assumes that p and q have same length and lie in the simplex of the dimension
		"""
		p = np.array(p)
		q = np.array(q)
		m = 0.5 * ( p+q )
		return 0.5 * ( mymath.KLD(p,m) + mymath.KLD(q,m) )


	@staticmethod
	def isInCircle(point, center, radius):
		"""
		Return a boolean value to tell wether coordinates are inside the
		circle with parameter center and radius
		
		Note: This function assumes we are dealing with numpy arrays
		"""
		return np.sqrt( np.sum((point-center)**2) ) < radius


	@staticmethod
	def log_add(x, y):
		"""
		Return the log sum of two numbers x and y already in log scale: log( exp(x) + exp(y) )
		in a floating point robust fashion
		"""
		logX = max(x,y)
		logY = min(x,y)
		negDiff = logY - logX
		# If the difference is too small, return the just the maximum
		if( negDiff < -20 ):
			return logX
		return logX + np.log( 1.0 + np.exp(negDiff) )


	@staticmethod
	def normalizeLines(mat):
		"""
		Normalize the lines of a Matrix
		(if a line contains a inf value, it will be scaled to 1 and 0 every where else)
		"""
		rowSums = np.sum(mat, axis=1)
		rowSums[rowSums == 0] = 1

		inf = np.where(mat == np.inf)
		mat[rowSums == np.inf] = 10e-200
		mat[inf] = 1
		
		rowSums[rowSums == np.inf] = 1

		return mat / np.tile(rowSums, (mat.shape[1], 1)).T


# ===================================================================================================================================
# ================= Sqlite3 utility functions
# ===================================================================================================================================

class sqlite3(object):

	@staticmethod
	def get_header(cursor):
		"""
		Return the header of a given cursor
		"""
		return [tuple[0] for tuple in cursor.description]

	@staticmethod
	def re_fn(expr, item):
		"""
		REGEXP function for Sqlite3
		"""
		#print "expr : %s" % expr
		#print "item : %s" % item 
		try:
			reg = re.compile(expr)
		except Exception as e:
			print ""
			print "Exception in re_fn: " + repr(e)
		return reg.search(item) is not None


# ===================================================================================================================================
# ================= Custom function from itertools package
# ===================================================================================================================================


class myitertools(object):

	@staticmethod
	def pairs(iterable):
		"""
		Implementation of combinaisons_with_replacement with r=2
		and do not iterate through paris of the same object
		e.g. with iterable=[1,2,3] returns: (1,2),(1,3),(2,3) but not (1,1),(2,2),(3,3),(2,1),...
		"""
		r = 2
		pool = tuple(iterable)
		n = len(pool)
		for indices in itertools.product(range(n), repeat=r):
			if sorted(indices) == list(indices) and indices[0] != indices[1]:
					yield tuple(pool[i] for i in indices)



# ===================================================================================================================================
# ================= Subclasses of Dict class
# ===================================================================================================================================


class MultiDict(dict):
    """
    Implementation of perl's autovivification feature
    """
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


class TwoWayDict(dict):
	"""
	Two way dictionary
	"""
	def __len__(self):
		return dict.__len__(self) / 2

	def __setitem__(self, key, value):
		dict.__setitem__(self, key, value)
		dict.__setitem__(self, value, key)


# ===================================================================================================================================
# ================= Printer utility class
# ===================================================================================================================================

class Printer(object):
	"""
	Clever printer
	"""

	def __init__(self, message="", delta=10000, verbose=True):
		self.counter = 0
		self.message = message
		self.delta = delta
		self.verbose = verbose
		self.start_time = time.time()

	def start(self, message=""):
		"""
		Reinitialize the time and the counter
		"""
		if self.verbose:
			stdout.write(message)
			stdout.flush()
			self.start_time = time.time()
			self.counter = 0

	def write(self, message=None):
		"""
		Write a line
		"""
		if self.verbose:
			if self.counter%self.delta==0: 
				if message == None:
					message = self.message

				m = (time.time()-self.start_time)/60
				stdout.write( "\r%d %s in %d min %d sec" % ( self.counter, message, m, (60*m)%60 ) )
				stdout.flush()
			self.counter+=1

	def stop(self, message=" done."):
		if self.verbose:
			stdout.write(message+"\n")
			stdout.flush()



class mystdout(object):

	last_write = time.time()
	write_interval = 0.1

	last_time = time.time()
	last_flush = time.time()

	# Indicate printing behavior
	verbose = False

	@staticmethod
	def write(string, progress=0, progress_max=1, show_percentage=True, show_progress=True, show_time=True, progress_len=35, message_len=70, ln=False):
		"""
		Dynamic printer with progress bar
		"""
		inter = time.time() - mystdout.last_write
		if mystdout.verbose and (inter > mystdout.write_interval or ln):
			mystdout.last_write = time.time()
			n = np.round(float(progress)*progress_len/progress_max)
			raw_s  = "\r" + string 													# Write string at start of line
			raw_s += " "*(message_len - len(string))								# Padding until max_length 	
			if show_progress:
				raw_s += "[" + "#"*n 												# Write progress bar
				raw_s += " "*(progress_len-n) + "]"									# Progress bar padding
				if show_percentage:
					raw_s +=  " {0:.0f}%".format(100*float(progress)/progress_max)	# Add percentage
				raw_s += " in %d sec" % mystdout.time_interval(update=ln)			# Show time
			raw_s += "\n"*ln 														# Add a \n if wanted
			stdout.write(raw_s)
			if time.time()-mystdout.last_flush > 0.1 or ln:
				stdout.flush()
				mystdout.last_flush = time.time()


	@staticmethod
	def time_interval(update=False):
		t = time.time() - mystdout.last_time
		if update:
			mystdout.last_time = time.time()
		return t

	@staticmethod
	def setVerbose(boolean):
		mystdout.verbose = boolean

	@staticmethod
	def setWriteInterval(interval):
		mystdout.write_interval = interval

	@staticmethod
	def resetLastWrite():
		mystdout.last_write = time.time() - mystdout.write_interval



