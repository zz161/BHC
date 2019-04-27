import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='BayesianClustering',  
     version='0.1',
     scripts=['bhc','node'] ,
     author="Azucena Morales and Alan Zhou",
     author_email="lm348@duke.edu and zz161@duke.edu",
     description="Bayesian Hierachichal Clustering using Heller's article",
     url="https://github.com/AzucenaMV/STAT-663",
     packages=setuptools.find_packages(['numpy','itertools','scipy','functools']),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
