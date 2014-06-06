import re
import numpy as np
from scipy import sparse
from sys import stdout
import time
import tables


# ===================================================================================================================================
# ================= arXiv database utility functions
# ===================================================================================================================================

def make_query_condition(start_date, end_date, categories):
	assert not bool(re.search("[^0-9\-:.\s]", start_date)), "Invalid character in start_date"
	assert not bool(re.search("[^0-9\-:.\s]", end_date)), "Invalid character in end_date"
	conditions = []
	for cat in categories:
		cat = cat.strip('"').strip("'")
		conditions.append("categories LIKE '%{}.%'".format(cat))
	cat_condition = '('+' OR '.join(conditions)+')'
	return "updated_at > '%s' AND updated_at < '%s' AND %s" % (start_date, end_date, cat_condition)


# ===================================================================================================================================
# ================= pyTables utility functions
# ===================================================================================================================================

def store_sparse_mat(mat, name, h5file, group):
	msg = "This code only works for csr matrices"
	assert(mat.__class__ == sparse.csr.csr_matrix), msg
	for par in ('data', 'indices', 'indptr', 'shape'):
		full_name = '%s_%s' % (name, par)
		try:
			n = getattr(group, full_name)
			n._f_remove()
		except AttributeError:
			pass
		arr = np.array(getattr(mat, par))
		atom = tables.Atom.from_dtype(arr.dtype)
		shape = arr.shape
		ds = h5file.create_carray(group, full_name, atom, shape)
		ds[:] = arr

def load_sparse_mat(name, h5file, group):
	pars = []
	for par in ('data', 'indices', 'indptr', 'shape'):
		pars.append(getattr(group, '%s_%s' % (name, par)).read())
	m = sparse.csr_matrix(tuple(pars[:3]), shape=pars[3])
	return m

def store_carray(arr, name, h5file, group):
	try:
		n = getattr(group, name)
		n._f_remove()
	except AttributeError:
		pass
	atom = tables.Atom.from_dtype(arr.dtype)
	shape = arr.shape
	ds = h5file.create_carray(group, name, atom, shape)
	ds[:] = arr



# ===================================================================================================================================
# ================= Printer utility class
# ===================================================================================================================================


class mystdout(object):

	last_write = time.time()
	# Maximum interval between two prints
	write_interval = 0.1
	# Indicate printing behavior
	verbose = True

	last_time = time.time()
	last_flush = time.time()

	@staticmethod
	def write(string, progress=0, progress_max=1, show_percentage=True, show_progress=True, show_time=True, progress_len=35, message_len=70, ln=False):
		"""
		Dynamic printer with progress bar
		"""
		inter = time.time() - mystdout.last_write
		if mystdout.verbose and (inter > mystdout.write_interval or ln):
			mystdout.last_write = time.time()
			n = np.round(float(progress)*progress_len/progress_max)
			raw_s  = "\r" + string                                                  # Write string at start of line
			raw_s += " "*(message_len - len(string))                                # Padding until max_length  
			if show_progress:
				raw_s += "[" + "#"*n                                                # Write progress bar
				raw_s += " "*(progress_len-n) + "]"                                 # Progress bar padding
				if show_percentage:
					raw_s +=  " {0:.0f}%".format(100*float(progress)/progress_max)  # Add percentage
				raw_s += " in %d min" % (mystdout.time_interval(update=ln)/60)      # Show time
			raw_s += "\n"*ln                                                        # Add a \n if wanted
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



