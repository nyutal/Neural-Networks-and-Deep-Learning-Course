# Created by @nyutal on 06/05/2020
from setuptools import setup, find_packages

setup_requires = [
    'pytest-runner',
]

install_requires = [
    'numpy',
]


tests_require = [
    'pytest',
    'pytest-cov',
]

setup(
    name='coursera_nn_course_1',
    description='implementaion for the first nn course by Andrew Eng',
    version='0.0.2',
    packages=find_packages('./src'),
    package_dir={'': 'src'},
    author='Nadav Yutal',
    author_mail='yutal.nadav@gmail.com',
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
)
