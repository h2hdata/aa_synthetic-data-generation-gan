

def _scatter_plot(data,x, y):
	"""
	   Function to create a scatter plot of one column versus
	   another.

	   Returns:
	   --------
	   scatter plot between x and y.

	"""
	ax = data.plot(x=x, y=y, kind='scatter')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_title(x+" versus "+y)
	plt.draw()
	plt.pause(0.01)
	# raw_input("Press enter to continue")

def _histogram_plot(data,x, y):
	"""
	   Function  to  create  a  histogram  plot of one column 
	   versus another.

	   Returns:
	   --------
	   histogram plot between x and y.

	"""
	ax = data.plot(x=x, y=y, kind='hist')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_title(x+" versus "+y)
	plt.draw()
	plt.pause(0.01)
	# raw_input("Press enter to continue")

def _box_plot(data,x, y):
	"""
	   Function  to  create  a  box plot of one column versus 
	   another.

	   Returns:
	   --------
	   box plot between x and y.

	"""
	ax = data.plot(x=x, y=y, kind='box')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_title(x+" versus "+y)
	plt.draw()
	plt.pause(0.01)
	# raw_input("Press enter to continue")

def _bar_chart(data,x):
	"""
	   Function  to  create  a bar chart of one column versus 
	   another.

	   Returns:
	   --------
	   bar chart.

	"""
	if x is not None:
		ax = data.groupby(x).count().plot(kind='bar')
		ax.set_xlabel(x)
		ax.set_title(x)
		plt.draw()
		plt.pause(0.01)
		# raw_input("Press enter to continue")
	else:
		ax = data.plot(kind='bar')
		plt.draw()
		plt.pause(0.01)
		# raw_input("Press enter to continue")




 
