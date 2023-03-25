Required Changes:
Line 12 of neural.py, found in mlrose library:
Change 'from sklearn.externals import six' to simply 'import six'. 
On my end, this fixed an ImportError encountered when otherwise trying to import mlrose.
