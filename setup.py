import os
import re
from setuptools import setup, find_packages
from setuptools.command.install import install


require_file = 'requirements.txt'
package_name = "deeperband"

verstrline = open(os.path.join(package_name, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(package_name))

with open(require_file) as f:
    requirements = [r.split()[0] for r in f.read().splitlines()]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Li Jun',
    author_email='ljcj007@ysu.edu.cn',
    url='https://github.com/ljcj007/DeeperBand',
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(package_name)
        ]
    },
)
