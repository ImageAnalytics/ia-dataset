from setuptools import setup

with open("requirements.txt") as fp:
    reqs = fp.readlines()

setup(
    name='ia_dataset',
    version='1.0.0',
    packages=[
        'ia_dataset',
    ],
    url='',
    license='',
    author='Ross Marchant',
    author_email='ross.g.marchant@gmail.com',
    description='Visage machine learning dataset management',
    install_requires=reqs
)
