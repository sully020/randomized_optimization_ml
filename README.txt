Download main.py, and ensure that all imports are installed. Ensure diabetes.csv is located in the same directory as main.py.

Required Changes:
Line 12 of 'neural.py', found in mlrose library:
Change 'from sklearn.externals import six' to simply 'import six'. 
This fixes an ImportError encountered when otherwise trying to import mlrose.

Additionally, add:
```
import warnings 
warnings.simplefilter('ignore')
```
to 'activation.py' of mlrose in order to suppress unneccessary warning spam.